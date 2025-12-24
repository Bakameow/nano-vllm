import torch
from nanovllm.utils.context import get_context, set_context, reset_context
from .base import BaseAttnBackend
from typing import List, Tuple
from nanovllm.engine.sequence import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


class FlashAttnBackend(BaseAttnBackend):
    def __init__(self):
        self.block_size = 256
        
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
        context = get_context()
        if context.is_prefill:
            o = flash_attn_varlen_func(q, k, v,
                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                    # softmax_scale=self.scale, 
                    causal=True, block_table=context.block_tables)
        else:
            o = flash_attn_with_kvcache(q.unsqueeze(1), k, v,
                    cache_seqlens=context.context_lens, block_table=context.block_tables, 
                    # softmax_scale=self.scale,
                    causal=True)
        return o

    def prepare_prefill(self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self._prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, attn_backend=self)
        return input_ids, positions

    def _prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self._prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, attn_backend=self)
        return input_ids, positions
    
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        pass

    def prepare_for_capture(self, seqs: List[Sequence]) -> None:
        pass

    def prepare_for_replay(self, seqs: List[Sequence]) -> None:
        pass