from typing import TypedDict, Optional
from enum import Enum
import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

class AttentionBackend(Enum):
    FLASH_ATTENTION = "flash"
    FLEX_ATTENTION = "flex"

BACKEND = AttentionBackend.FLASH_ATTENTION

def set_backend(backend: AttentionBackend):
    global BACKEND
    BACKEND = backend

def get_backend():
    return BACKEND

class AttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flex Attention.
    """
    cumulative_seqlens_q: Optional[torch.LongTensor]
    cumulative_seqlens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]
    flex_attn_block_mask: Optional[torch.Tensor]
    flex_attn_input_pos: Optional[torch.Tensor]
    flex_attn_batch_idx: Optional[torch.Tensor]

def create_causal_blockmask(B, L, page_size: int, device: str) -> BlockMask:
    """A minimal, unoptimized causal block mask creation function"""

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return create_block_mask(causal, B=B, H=None, Q_LEN=L, KV_LEN=L, BLOCK_SIZE=page_size, device=device)
