import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_tokens = logits.argmax(dim=-1)
        safe_temp = torch.where(temperatures > 0, temperatures, 1.0).unsqueeze(dim=1)
        logits = logits.float().div_(safe_temp)
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(temperatures > 0, sample_tokens, greedy_tokens)
