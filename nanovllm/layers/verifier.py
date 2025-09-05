import torch
import json
import os
from pathlib import Path
import xxhash
from functools import lru_cache
import numpy as np
from nanovllm.engine.sequence import Sequence

class LogitsSaver:
    """保存同一 seq_id 的 sequence logits 到文件的工具类"""
    
    def __init__(self, model: str, save_dir: str = "tmp", ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.model = model
    
    def _convert_tensor_to_list(self, logits: torch.Tensor):
        """tensor 转换函数"""
        return logits.cpu().numpy().tolist()

    def _compute_hash(self, token_ids: list[int]):
        """
        计算完整 prompt token 序列的哈希值
        """
        h = xxhash.xxh64()
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def add_logits(self, seq: Sequence, logits: torch.Tensor):
        seq.logits_list.append(self._convert_tensor_to_list(logits))

    def save_logits(self, seq: Sequence):
        """
        保存指定 seq_id 的 logits 到文件
        
        Args:
            seq: 序列
            logits: 对应的logits张量
        """
        hash_id = self._compute_hash(seq.prompt_token_ids)
        filename = self.save_dir / f"{self.model}/seq_{hash_id}.json"
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
        data["tokens"].append(np.argmax(seq.logits_list[-1]).item())
        # assert data["tokens"][0] == np.argmax(seq.logits_list[0]).item()
        data["logits"].extend(seq.logits_list)
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

class LogitsVerifier:
    def __init__(self,  target_model: str, draft_model: str, save_dir: str = "tmp",):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.target_model = target_model
        self.draft_model = draft_model
        # self._cache = {}

    def _get_cache_data(self, filename: Path):
        if filename not in self._cache:
            return self._cache[filename]
        return None

    @staticmethod
    @lru_cache(maxsize=10)
    def _cache_json(path: Path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _compute_hash(self, token_ids: list[int]):
        """
        计算完整 prompt token 序列的哈希值
        """
        h = xxhash.xxh64()
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def dump_logits(self, seq: Sequence, target_logits: torch.Tensor, draft_logits: torch.Tensor):
        hash_id = self._compute_hash(seq.prompt_token_ids)
        file_dir = self.save_dir / f"{self.draft_model}"
        file_dir.mkdir(parents=True, exist_ok=True)
        with open(file_dir / f"seq_{hash_id}_draft.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(draft_logits.cpu().numpy().tolist(), ensure_ascii=False) + "\n")
        with open(file_dir / f"seq_{hash_id}_target.jsonl", 'a', encoding='utf-8') as f:
            f.write(json.dumps(target_logits.cpu().numpy().tolist(), ensure_ascii=False) + "\n")

    def verify_logits(self, seq: Sequence, logits: torch.Tensor):
        hash_id = self._compute_hash(seq.prompt_token_ids)
        filename = self.save_dir / f"{self.target_model}/seq_{hash_id}.json"
        if not filename.exists():
            print(f"file not found for {filename}")
            return logits
            # raise FileNotFoundError(f"File {filename} not found")
            
        target = self._cache_json(filename)

        if target["tokens"][seq.num_completion_tokens] == logits.argmax(dim=-1).item():
            print(f"logits match for seq {hash_id} at pos {seq.num_completion_tokens}")
            return logits
        else:
            target_logits = torch.tensor(target["logits"][seq.num_completion_tokens])
            print(f"logits mismatch for seq {hash_id} at pos {seq.num_completion_tokens}")
            self.dump_logits(seq, target_logits, logits)
            return target_logits.to(logits.device)