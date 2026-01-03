from dataclasses import dataclass, replace
import torch
from typing import TypedDict, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from nanovllm.attention import BaseAttnBackend
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper
    )

@dataclass
class Context:
    # shared across backends
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # flashattn related
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    context_lens: torch.Tensor | None = None
    # flashinfer related
    paged_kv_indices: torch.Tensor | None = None
    paged_kv_last_page_len: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    num_qo_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    page_size: Literal[1, 256] = 256
    pos_encoding_mode: str = 'NONE'
    q_data_type: str | torch.dtype = 'float16'
    kv_data_type: str | torch.dtype | None = None
    # masking
    custom_mask: torch.Tensor | None = None
    packed_custom_mask: torch.Tensor | None = None
    # control message
    initialized: bool = False
    attn_backend: "BaseAttnBackend | None" = None
    wrapper: "BatchDecodeWithPagedKVCacheWrapper | BatchPrefillWithPagedKVCacheWrapper | None" = None

    def __post_init__(self):
        pass

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
        *,
        is_prefill,
        **kwargs):
    global _CONTEXT
    # _CONTEXT = Context(is_prefill, **kwargs)
    _CONTEXT = replace(_CONTEXT, is_prefill=is_prefill, **kwargs)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
