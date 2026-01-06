import torch
from typing import TYPE_CHECKING

from .base import BaseAttnBackend
from .fa import FlashAttnBackend
from .fi import FlashInferBackend
from nanovllm.config import ModelConfig

if TYPE_CHECKING:
    from nanovllm.config import ModelConfig

def create_attn_backend(
        config: ModelConfig,
        backend: str,
        page_table: torch.Tensor
) -> BaseAttnBackend:
    match backend:
        case "fa2":
            return FlashAttnBackend()
        case "flashinfer":
            return FlashInferBackend(config, page_table)
        case _:
            raise ValueError(f"Unknown attention backend: {backend}")

__all__ = ["BaseAttnBackend", "create_attn_backend", "FlashAttnBackend"]
