import torch
from torch import nn
import json
import os
from pathlib import Path
import xxhash
import numpy as np
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.verifier import LogitsSaver, LogitsVerifier

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        # 一个 batch_size 内的所有 logits，除以各自请求的temperature
        logits.div_(temperatures.unsqueeze(dim=1))
        # softmax归一化
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        # 使用gumbel-max trick进行采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)

class SpecSampler(nn.Module):
    def __init__(self, model: str, logits_save_dir: str = "tmp"):
        super().__init__()
        self.model : str = os.path.basename(model)
        self.logits_saver = LogitsSaver(self.model, logits_save_dir)
        #[MODIFY]
        self.save_logits = True
        

    def forward(self, seqs: list[Sequence], logits: torch.Tensor, temperatures: torch.Tensor):
        
        logits = logits.to(torch.float)
        
        # 如果需要保存logits
        if self.save_logits:
            for i, seq in enumerate(seqs):
                if i < logits.shape[0]:  # 确保索引有效
                    self.logits_saver.add_logits(seq, logits[i])
                if seq.num_completion_tokens == seq.max_tokens-1:
                    self.logits_saver.save_logits(seq)
                
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)

class DraftSampler(nn.Module):
    def __init__(self, model: str, target_model: str, logits_save_dir: str = "tmp"):
        super().__init__()
        self.model : str = os.path.basename(model)
        self.target_model : str = os.path.basename(target_model)
        self.logits_verifier = LogitsVerifier(self.target_model, self.model, logits_save_dir)
        #[MODIFY]
        self.save_logits = True


    def forward(self, seqs: list[Sequence], logits: torch.Tensor, temperatures: torch.Tensor):
        
        logits = logits.to(torch.float)

        if self.save_logits:
            for i, seq in enumerate(seqs):
                if i < logits.shape[0]:  # 确保索引有效
                    logits[i] = self.logits_verifier.verify_logits(seq, logits[i])
                
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)