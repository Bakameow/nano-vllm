from dataclasses import dataclass
import torch
from typing import TYPE_CHECKING, List, Literal
from nanovllm.engine.sequence import Sequence

if TYPE_CHECKING:
    from nanovllm.attention import BaseAttnBackend, BaseAttnMetadata
    from nanovllm.engine.sequence import Sequence

class Batch:
    def __init__(self, *, seqs: List[Sequence], phase: Literal["prefill", "decode"]):
        self.seqs = seqs
        self.phase: Literal["prefill", "decode"] = phase
        # these fields should be set by scheduler
        # self.input_ids: torch.Tensor
        # self.out_loc: torch.Tensor
        # self.padded_reqs: List[Sequence]  # may contain some dummy reqs for padding
        # this field should be set by attention backend
        # self.attn_metadata: BaseAttnMetadata

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    # @property
    # def size(self) -> int:
    #     return len(self.reqs)
    # @property
    # def padded_size(self) -> int:
    #     return len(self.padded_reqs)

@dataclass
class Context:
    block_tables: torch.Tensor | None = None
    attn_backend: "BaseAttnBackend | None" = None
    attn_metadata: "BaseAttnMetadata | None" = None
    batch : Batch | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(block_tables=None, attn_backend=None, batch=None, attn_metadata=None):
    global _CONTEXT
    _CONTEXT = Context(block_tables=block_tables, attn_backend=attn_backend, batch=batch, attn_metadata=attn_metadata)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()