import torch
from torch import nn
import json
import os
from pathlib import Path
import xxhash
import numpy as np
from nanovllm.engine.sequence import Sequence

class LogitsSaver:
    """保存同一 seq_id 的 sequence logits 到文件的工具类"""
    
    def __init__(self, save_dir: str = "logits_output"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def _convert_tensor_to_list(self, logits: torch.Tensor):
        """tensor 转换函数"""
        return logits.cpu().numpy().tolist()

    def add_logits(self, seq: Sequence, logits: torch.Tensor):
        """
        添加logits到序列中
        """
        seq.logits_list.append(self._convert_tensor_to_list(logits))

    def _compute_hash(self,token_ids: list[int]):
        """
        计算完整 prompt token 序列的哈希值
        """
        h = xxhash.xxh64()
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def save_logits(self, model: str, seq: Sequence):
        """
        保存指定 seq_id 的 logits 到文件
        
        Args:
            seq: 序列
            logits: 对应的logits张量
        """
        hash_id = self._compute_hash(seq.prompt_token_ids)
        filename = self.save_dir / f"{model}/seq_{hash_id}.json"
        filename.parent.mkdir(exist_ok=True)
        # 检查文件是否已存在
        if filename.exists():
            print("dumped sequence of hash_id", hash_id)
            return

        data = {
            "hash_id": hash_id,
            "tokens": [],
            "logits": []
        }
        data["tokens"].extend(seq.completion_token_ids)
        data["logits"].extend(seq.logits_list)
        
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
    def __init__(self, model: str, logits_save_dir: str = "logits_output"):
        super().__init__()
        self.logits_saver = LogitsSaver(logits_save_dir)
        #[MODIFY]
        self.save_logits = True
        self.model : str = os.path.basename(model)
        self.pending_tasks = []

    def forward(self, seqs: list[Sequence], logits: torch.Tensor, temperatures: torch.Tensor):
        
        logits = logits.to(torch.float)
        
        # 如果需要保存logits
        if self.save_logits:
            for i, seq in enumerate(seqs):
                if i < logits.shape[0]:  # 确保索引有效
                    self.logits_saver.add_logits(seq, logits[i])
                    # task = asyncio.create_task(
                    #     self.logits_saver.add_logits(seq, logits[i])
                    # )
                    # self.pending_tasks.append(task)
                print(seq.max_tokens)
                print(seq.num_completion_tokens)
                if seq.num_completion_tokens == seq.max_tokens-1:
                    self.logits_saver.save_logits(self.model, seq)
                
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)

    async def wait_for_pending_tasks(self):
        """等待所有待处理的任务完成"""
        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks)
            self.pending_tasks.clear()