import torch
from abc import ABC, abstractmethod
from typing import List, Tuple
from nanovllm.engine.sequence import Sequence

class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_prefill(self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def prepare_decode(self, seqs: List[Sequence]) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, seqs: List[Sequence]) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, seqs: List[Sequence]) -> None: ...

