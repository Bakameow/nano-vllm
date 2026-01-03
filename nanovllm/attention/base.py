import torch
from abc import ABC, abstractmethod
from typing import List, Tuple
from dataclasses import dataclass
from nanovllm.utils.context import Batch

@dataclass
class BaseAttnMetadata(ABC):
    # positions: torch.Tensor

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...

class BaseAttnBackend(ABC):
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor: ...

    @abstractmethod
    def prepare_prefill(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def prepare_decode(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...

