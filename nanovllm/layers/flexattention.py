import torch
from torch import nn
from typing import Optional, Unpack
import triton
import triton.language as tl
from nanovllm.layers.flashattention import store_kvcache
from nanovllm.layers.flashattention import flash_attn_varlen_func
from torch.nn.attention.flex_attention import (
    create_block_mask,
    noop_mask,
    flex_attention,
    _identity,
    _score_mod_signature,
    _mask_mod_signature
)
from nanovllm.utils.context import get_context
from nanovllm.utils.attention_utils import AttentionKwargs

PAGE_SIZE = 256
flex_attention = torch.compile(flex_attention, fullgraph=True)

class FlexAttention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # NOTE: 在 model_runner 中，会调用 allocate_kv_cache 为模型分配 KV Cache，并保存到 self.k_cache 和 self.v_cache 中
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        **kwargs : Unpack[AttentionKwargs],
    ):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 如果KV cache 不为空，则将K和V缓存到KV cache中
        # 这里的KV是添加过 Rotary Embedding 的
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flex_attention(
                    q.unsqueeze(1),
                    k_cache,
                    v_cache,
                    block_mask=context.flex_attn_block_mask,
                    # score_mod=paged_score_mod,
                    enable_gqa=True,
                    )
        o = o.view(-1, self.num_heads * self.head_dim)
        return o

