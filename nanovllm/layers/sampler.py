import torch
from torch import nn
import json
import os
from pathlib import Path
from nanovllm.engine.sequence import Sequence

class LogitsSaver:
    """保存同一 seq_id 的 sequence logits 到文件的工具类"""
    
    def __init__(self, save_dir: str = "logits_output"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_logits(self, seq_id: int, token_id: int, logits: torch.Tensor):
        """
        保存指定 seq_id 的 logits 到文件
        
        Args:
            seq_id: 序列ID
            token_id: 当前token的ID
            logits: 对应的logits张量
        """
        filename = self.save_dir / f"seq_{seq_id}_logits.json"
        
        # 将logits转换为numpy数组，然后转换为列表以便JSON序列化
        logits_list = logits.cpu().numpy().tolist()
        
        # 检查文件是否已存在
        if filename.exists():
            # 如果文件存在，读取现有内容并更新
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            # 如果文件不存在，创建新的数据结构
            data = {
                "seq_id": seq_id,
                "tokens": [],
                "logits": []
            }
        
        # 添加新的token_id和logits
        data["tokens"].append(token_id)
        data["logits"].append(logits_list)
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)

class SpecSampler(nn.Module):
    def __init__(self, logits_save_dir: str = "logits_output"):
        super().__init__()
        self.logits_saver = LogitsSaver(logits_save_dir)
        self.save_logits = False

    def forward(self, seqs: list[Sequence], logits: torch.Tensor, temperatures: torch.Tensor):
        
        logits = logits.to(torch.float)
        
        # 如果需要保存logits
        if self.save_logits:
            for i, seq in enumerate(seqs):
                if i < logits.shape[0]:  # 确保索引有效
                    self.logits_saver.save_logits(seq.seq_id, seq.last_token, logits[i])
        
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)