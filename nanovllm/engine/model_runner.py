import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.utils.log import log_info, log_error, log_debug
import random
import copy
from nanovllm.utils.attention_utils import (
    set_backend,
    get_backend,
    AttentionBackend,
    create_causal_blockmask,
)
from torch.nn.attention.flex_attention import (
    BlockMask,
    _mask_mod_signature,
    noop_mask,
)
from typing import Optional

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        log_info(f"config: {config}")
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        set_backend(config.attention_backend)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            # 主进程创建共享内存
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                # 进程同步屏障，所有进程都会在这里等待，直到大家都到达这一步，才会继续往下执行
                dist.barrier()
            else:
                dist.barrier()
                # 从进程（副 Rank）连接主进程（主 Rank）的共享内存
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        由从 Rank 调用，从共享内存中读取方法名和参数，并返回；

        Returns:
            tuple: 方法名和参数
        """
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        由主 Rank 调用，将方法名和参数写入共享内存；

        Args:
            method_name (str): 方法名
            *args: 方法参数

        Returns:
            None
        """
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 计算单个 Rank 保存的 KV head 头数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # [Q]这里在计算什么？
        # [A]计算一个 KV Cache Block 占用的字节数，因为是 K 和 V 的缓存，所以需要缓存两份；hf_config.num_hidden_layers 是Decoder堆叠的层数，每层的注意力都需要保存 KV Cache；self.block_size 是每个 KV Cache Block 的可以保存的 token 数；num_kv_heads 是 GQA 注意力的 head 数；hf_config.head_dim 是每个头的维度，也是每个 KV Cache 隐藏层的维度；hf_config.torch_dtype.itemsize 是每个元素的字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # NOTE: KV Cache 的形状为 [2, 隐藏层层数, KV Cache 块数, KV Cache Block 大小, num_kv_heads, head_dim]，2 表示 K 和 V 的缓存
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        # NOTE: 增加倒置页表，用于计算 Flex Attention 的 Block Mask
        if get_backend() == AttentionBackend.FLEX_ATTENTION:
            self.inverted_block_table = -torch.ones((config.max_num_seqs, config.num_kvcache_blocks),dtype=torch.int32)
            self.block_mask = create_causal_blockmask(B=config.max_num_seqs, L=config.max_model_len, page_size=self.block_size, device="cuda")
        else:
            self.inverted_block_table = None
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """
        将 seqs 序列的 block_table 转换为 tensor，并返回；

        Args:
            seqs (list[Sequence]): 要转换的 seq 序列

        Returns:
            torch.Tensor: 转换后的 block_table tensor
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        # 将每个 seq 的 block_table 填充到 max_len 长度，不足的用 -1 填充
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_evicted_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        # return self.prepare_block_tables(seqs)
        def remove_random_block(block_table: list[int]):                
            remove_count = max(1, len(block_table) // 4)
            indices_to_remove = set(random.sample(range(len(block_table)), remove_count))
            
            block_table = [x for i, x in enumerate(block_table) if i not in indices_to_remove]
            return block_table

        selected_block_tables = []
        for seq in seqs:
            if seq.is_compressed == SequenceStatus.COMPRESSED :
                assert seq.num_blocks >= 8
                selected_block_tables.append(seq.compressed_block_table+seq.block_table[8:])
            elif seq.num_blocks < 8:
                selected_block_tables.append(seq.block_table)
            else:
                seq.compressed_block_table = copy.deepcopy(seq.block_table)
                seq.compressed_block_table = remove_random_block(seq.compressed_block_table)
                selected_block_tables.append(seq.compressed_block_table)
                seq.is_compressed = SequenceStatus.COMPRESSED
        
        max_len = max(len(block_table) for block_table in selected_block_tables)
        block_tables = [block_table + [-1] * (max_len - len(block_table)) for block_table in selected_block_tables]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        # 累计query、key序列长度
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            # 为各个输入的 token id 生成位置索引
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            # 记录本 batch 里最长的 query/key 序列长度。
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                # NOTE: 将各个 Block Size 规模的 token 对应的 KV Cache 的所在的 slot 下标映射到 slot_mapping 中，在调用 Attention 时会使用 triton kernel 将 KV 缓存到各个层的 kv_cache 中
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_flash_attention(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for idx, seq in enumerate(seqs):
            # 获取最后一个 token 的 id 和位置索引
            input_ids.append(seq.last_token)
            positions.append(len(seq)-1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # log_debug(f"block_tables: {block_tables}")
        # log_debug(f"slot_mapping: {slot_mapping}")
        # log_debug(f"positions: {positions}")
        # log_debug(f"context_lens: {context_lens}")
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def gen_offset(off: torch.Tensor):
        """Generates an offset function.

        Args:
            off: Offset tensor.
        """
        def offset(B, H, M, N):
            # NOTE: B = batch_idx, H = head_idx, M = q_idx, N = kv_idx
            return M + off[B] >= N
        return offset

    def get_logical_kv_idx(self, physical_batch_idx: torch.Tensor, physical_kv_idx: torch.Tensor, batch_idx: torch.Tensor):
        logical_batch_idx = batch_idx[physical_batch_idx]
        physical_kv_block = physical_kv_idx // self.block_size
        physical_kv_offset = physical_kv_idx % self.block_size
        logical_block_idx = self.inverted_block_table[logical_batch_idx, physical_kv_block]
        logical_kv_idx = logical_block_idx * self.block_size + physical_kv_offset
        is_valid = logical_block_idx >= 0
        safe_logical_kv_idx = logical_kv_idx.clamp(min=0)
        return is_valid, safe_logical_kv_idx

    def get_mask_mod(self, mask_mod: Optional[_mask_mod_signature], batch_idx: torch.Tensor) -> _mask_mod_signature:
        """
        Converts a mask_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            mask_mod (_mask_mod_signature): mask_mod based on the logical block index.
        """
        if mask_mod is None:
            mask_mod = noop_mask

        def new_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ):
            is_valid, safe_logical_kv_idx = self.get_logical_kv_idx(b, physical_kv_idx, batch_idx)
            return torch.where(is_valid, mask_mod(b, h, q_idx, safe_logical_kv_idx), False)

        return new_mask_mod

    def convert_logical_block_mask(
        self,
        block_mask: BlockMask,
        block_tables: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> BlockMask:
        B, H, ROWS, MAX_BLOCKS_IN_COL = block_mask.kv_indices.shape
        if block_mask.BLOCK_SIZE[1] != self.block_size:
            raise RuntimeError(
                f"Expect block_mask has the same column block size as page_sizebut got size={block_mask.BLOCK_SIZE[1]} and size={self.block_size}"
            )
        device = block_mask.kv_num_blocks.device
        if batch_idx is None:
            batch_idx = torch.arange(B, device=device)

        assert batch_idx.ndim == 1, "batch_idx must be a 1D tensor"
        assert batch_idx.shape[0] == B, "batch_idx must have the same shape as block_mask"
        assert B <= self.config.max_num_seqs, "batch_idx must be less than or equal to max_num_seqs"
        
        def transform(num_blocks, indices):
            """
            transform the block mask from [B, H, num_q_blocks, num_logical_kv_blocks]
            to [B, H, num_q_blocks, num_physical_kv_blocks]

            kv_num_blocks: [B, H, num_q_blocks] -> unchanged
            kv_indices: [B, H, num_q_blocks, num_logical_kv_blocks] -> [B, H, num_q_blocks, num_physical_kv_blocks]
            """
            if num_blocks is None:
                return None, None
            new_kv_num_blocks = num_blocks.clone()
            new_kv_indices = torch.zeros((B, H, ROWS, self.config.num_kvcache_blocks), dtype=torch.int32, device=device)
            new_kv_indices[:, :, :, :MAX_BLOCKS_IN_COL] = (
                torch.gather(block_tables, 1, indices.view(B, -1).to(torch.int64)).view(block_mask.kv_indices.shape).to(torch.int32)
            )
            return new_kv_num_blocks, new_kv_indices

        new_kv_num_blocks, new_kv_indices = transform(block_mask.kv_num_blocks, block_mask.kv_indices)
        new_full_kv_num_blocks, new_full_kv_indices = transform(block_mask.full_kv_num_blocks, block_mask.full_kv_indices)
        new_mask_mod = self.get_mask_mod(block_mask.mask_mod, batch_idx)
        seq_lengths = (block_mask.seq_lengths[0], self.config.num_kvcache_blocks * self.block_size)
        return BlockMask.from_kv_blocks(
            new_kv_num_blocks,
            new_kv_indices,
            new_full_kv_num_blocks,
            new_full_kv_indices,
            block_mask.BLOCK_SIZE,
            new_mask_mod,
            seq_lengths=seq_lengths,
        )

    def prepare_flex_attention(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:

        # NOTE: this function is entirely in logical space
        def causal_offset(off: torch.Tensor):
            def offset(b, h, q_idx, kv_idx):
                return q_idx + off[b] >= kv_idx
            return offset
        
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        # NOTE: for flex attention
        batch_idx = []
        inverted_block_table = []
        for idx, seq in enumerate(seqs):
            # 获取最后一个 token 的 id 和位置索引
            input_ids.append(seq.last_token)
            batch_idx.append(idx)
            positions.append(len(seq)-1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            inverted_block_table.append(seq.inverted_page_table)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        batch_idx = torch.tensor(batch_idx, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        self.inverted_block_table[batch_idx, :] = torch.tensor(inverted_block_table, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        assert batch_idx.ndim == 1, "batch_idx must be 1D"
        assert positions.ndim == 1, "positions must be 1D"
        (B,) = batch_idx.shape

        block_mask = self.block_mask
        logical_block_idx = positions // self.block_size    # [B]
        kv_num_blocks = block_mask.kv_num_blocks[batch_idx, :, logical_block_idx].view(B, 1, 1)
        kv_indices = block_mask.kv_indices[batch_idx, :, logical_block_idx].view(B, 1, 1, -1)
        full_kv_num_blocks, full_kv_indices = None, None
        if block_mask.full_kv_num_blocks is not None:
            full_kv_num_blocks = block_mask.full_kv_num_blocks[batch_idx, :, logical_block_idx].view(B, 1, 1)  # noqa
            full_kv_indices = block_mask.full_kv_indices[batch_idx, :, logical_block_idx].view(B, 1, 1, -1)  # noqa
        seq_length = (1, block_mask.seq_lengths[1])
        mask = BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            BLOCK_SIZE=block_mask.BLOCK_SIZE,
            mask_mod=causal_offset(positions),
            seq_lengths=seq_length,
        )
        mask = self.convert_logical_block_mask(mask, block_tables, batch_idx)

        log_debug(f"block_tables: {block_tables}")
        log_debug(f"slot_mapping: {slot_mapping}")
        log_debug(f"context_lens: {context_lens}")
        log_debug(f"batch_idx: {batch_idx}")
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, flex_attn_block_mask=mask)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        if get_backend() == AttentionBackend.FLASH_ATTENTION:
            log_info("prepare_decode: flash_attention")
            input_ids, positions = self.prepare_flash_attention(seqs)
        elif get_backend() == AttentionBackend.FLEX_ATTENTION:
            log_info("prepare_decode: flex_attention")
            input_ids, positions = self.prepare_flex_attention(seqs)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        运行模型，计算 logits；

        Args:
            input_ids (torch.Tensor): 输入的 token id 序列；
            positions (torch.Tensor): 输入的 token 位置索引序列；
            is_prefill (bool): 是否是 prefill；

        Returns:
            next_token_logits (torch.Tensor): 下一个 token 的 logits；
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 如果batch_size过大，或处于 prefill 阶段，则直接计算 logits；
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        运行模型，根据 seqs 推理出每个 seq 的新 token

        Args:
            seqs (list[Sequence]): 要运行的 seq 序列
            is_prefill (bool): 是否是 prefill
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获模型计算图；

        Returns:
            None
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
