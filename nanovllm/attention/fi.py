import torch
from nanovllm.utils.context import get_context, set_context, reset_context, Context
from .base import BaseAttnBackend
from typing import List, Tuple
from nanovllm.engine.sequence import Sequence
from nanovllm.config import ModelConfig
from typing import TYPE_CHECKING
import math
from loguru import logger
if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )

def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))

class FlashInferBackend(BaseAttnBackend):
    def __init__(self, model_config: ModelConfig):
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )
        self.config = model_config
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8
        )
        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, device="cpu", pin_memory=True)
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is buggy, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
        )
        # NOTE: some hack to reuse the int_workspace_buffer
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer
        self.block_size = 1

        # temporarily doesn't support tensor parallelism
        self.qo_head_local = self.config.num_qo_heads
        self.kv_head_local = self.config.num_kv_heads

    @staticmethod
    def _initialize_metadata_once(context: Context) -> None:
        if context.initialized:
            return
        
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        logger.debug(context)
        context.initialized = True
        if isinstance(context.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            context.wrapper.plan(
                indptr=context.cu_seqlens_k,
                indices=context.paged_kv_indices,
                last_page_len=context.paged_kv_last_page_len,
                num_qo_heads=context.num_qo_heads,
                num_kv_heads=context.num_kv_heads,
                head_dim=context.head_dim,
                page_size=context.page_size,
                pos_encoding_mode=context.pos_encoding_mode,
                seq_lens=context.seq_lens,
                data_type=context.q_data_type,
                q_data_type=context.q_data_type,
                kv_data_type=context.kv_data_type,
                non_blocking=True,
            )
        else:
            context.wrapper.plan(
                qo_indptr=context.cu_seqlens_q,
                paged_kv_indptr=context.cu_seqlens_k,
                paged_kv_indices=context.paged_kv_indices,
                paged_kv_last_page_len=context.paged_kv_last_page_len,
                num_qo_heads=context.num_qo_heads,
                num_kv_heads=context.num_kv_heads,
                head_dim_qk=context.head_dim,
                page_size=context.page_size,
                pos_encoding_mode=context.pos_encoding_mode,
                seq_lens=context.seq_lens,
                q_data_type=context.q_data_type,
                kv_data_type=context.kv_data_type,
                non_blocking=True,
                causal=True,
            )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        context = get_context()
        self._initialize_metadata_once(context)
        # return context.wrapper.run(q, paged_kv_cache=(k.contiguous(), v.contiguous()))
        return context.wrapper.run(q, paged_kv_cache=(k, v))

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, device="cpu", pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def make_positions(self, device: torch.device, reqs: List[Sequence]) -> torch.Tensor:
        needed_size = sum(req.num_tokens - req.num_cached_tokens for req in reqs)
        indices_host = torch.empty(needed_size, dtype=torch.int32, device="cpu", pin_memory=True)
        offset = 0
        for req in reqs:
            length = req.num_tokens - req.num_cached_tokens
            torch.arange(
                req.num_cached_tokens,
                req.num_tokens,
                dtype=torch.int32,
                device="cpu",
                out=indices_host[offset : offset + length],
            )
            offset += length
        return indices_host.to(device, non_blocking=True)

    def prepare_prefill(self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        padded_size = len(seqs)
        seqlens_q = [len(seq) - seq.num_cached_tokens for seq in seqs]
        seqlens_k = [len(seq) for seq in seqs]
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        input_ids = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            # positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                logger.debug(f"seq.block_table={seq.block_table}")
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self._prepare_block_tables(seqs)

        seq_len_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = self.make_positions(device=torch.device("cpu"), reqs=seqs).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        logger.debug(f"slot_mapping={slot_mapping}")
        set_context(is_prefill=True,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    slot_mapping=slot_mapping,
                    block_tables=block_tables,
                    context_lens=None,
                    paged_kv_indices=slot_mapping,
                    paged_kv_last_page_len=self._get_ones_cpu(padded_size),
                    num_qo_heads=self.qo_head_local,
                    num_kv_heads=self.kv_head_local,
                    head_dim=self.config.head_dim,
                    page_size=1,
                    pos_encoding_mode='NONE',
                    q_data_type=self.config.dtype,
                    kv_data_type=self.config.dtype,
                    initialized=False,
                    wrapper=self.prefill_wrapper,
                    attn_backend=self)
        return input_ids, positions

    def _prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def _prepare_page_indices(self, seqs: list[Sequence]) -> torch.Tensor:
        page_indices = []
        for seq in seqs:
            page_indices.extend(seq.block_table[:seq.num_blocks])
        page_indices = torch.tensor(page_indices, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        seqlens_k = [len(seq) for seq in seqs]
        assert page_indices.size(0) == sum(seqlens_k), f"page_indices.size(0)={page_indices.size(0)}, sum(seqlens_k)={sum(seqlens_k)}"
        return page_indices

    def prepare_decode(self, seqs: list[Sequence]):
        padded_size = len(seqs)
        seqlens_q = [len(seq) - seq.num_cached_tokens for seq in seqs]
        seqlens_k = [len(seq) for seq in seqs]
        max_seqlen_q = max(seqlens_q)
        cpu_kwargs = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []
        for seq in seqs:
            seqlen = len(seq)
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            # seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            # cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)


        cu_seqlens_q.extend(range(1, padded_size + 1))
        
        seq_len_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        logger.debug(f"cu_seqlens_q={cu_seqlens_q}")
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        logger.debug(f"cu_seqlens_k={cu_seqlens_k}")
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        logger.debug(f"slot_mapping={slot_mapping}")
        indices = self._prepare_page_indices(seqs)
        set_context(is_prefill=False,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    slot_mapping=slot_mapping,
                    context_lens=None,
                    paged_kv_indices=indices,
                    paged_kv_last_page_len=self._get_ones_cpu(padded_size),
                    num_qo_heads=self.qo_head_local,
                    num_kv_heads=self.kv_head_local,
                    head_dim=self.config.head_dim,
                    page_size=1,
                    pos_encoding_mode='NONE',
                    q_data_type=self.config.dtype,
                    kv_data_type=self.config.dtype,
                    initialized=False,
                    wrapper=self.decode_wrappers,
                    attn_backend=self)
        return input_ids, positions
    
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        pass

    def prepare_for_capture(self, seqs: List[Sequence]) -> None:
        pass

    def prepare_for_replay(self, seqs: List[Sequence]) -> None:
        pass