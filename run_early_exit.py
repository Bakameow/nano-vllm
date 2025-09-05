import os
import json
from pathlib import Path
import xxhash
import numpy as np
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from collections import defaultdict
import tqdm

def compute_hash(token_ids: list[int]):
    """
    计算完整 prompt token 序列的哈希值
    """
    h = xxhash.xxh64()
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()

def read_jsonl(path: str) -> list:
    """读取 JSONL 文件并返回 list"""
    file_path = Path(path)
    
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")

    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data_list.append(json.loads(line))
    return data_list

def main():
    path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, early_exit=True)

    # sampling_params = SamplingParams(temperature=0, max_tokens=50)
    # prompt_list = read_jsonl("/root/nano-vllm/simple_question.jsonl")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
    prompt_list = read_jsonl("/root/nano-vllm/select_question.jsonl")
    for i in tqdm.tqdm(range(0,len(prompt_list),5)):
        batch = prompt_list[i:i+5]
        prompts = []
        for elem in batch:
            prompts.append(elem["turns"][0])
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            for prompt in prompts
        ]
        outputs = llm.generate(prompts, sampling_params)
        print(outputs)

def filter_jsonl(path: str, num: int):
    """过滤 JSONL 文件中的数据"""
    path : Path= Path(path)
    data_list = read_jsonl(path)
    data_dict = defaultdict(list)

    for item in data_list:
        data_dict[item["category"]].append(item)

    select_data_list = []
    for category, items in data_dict.items():
        select_data_list.extend(items[:num])
    select_path = path.parent / f"select_{path.name}"
    with open(select_path, "w", encoding="utf-8") as f:
        for item in select_data_list:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
    
def tokenize_prompt():
    prompt = "1+1="
    path = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    print(prompt)
    print(tokenizer.encode(prompt))

if __name__ == "__main__":
    main()
    # tokenize_prompt()
    # select_data_list = filter_jsonl("/root/nano-vllm/question.jsonl", 10)