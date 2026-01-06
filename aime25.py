import os
import json
import re
from tqdm import tqdm
from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from loguru import logger

def extract_answer(text):
    # Try different patterns to extract the answer
    patterns = [
        r"\\mathbf\{(.*?)\}",
        r"boxed\{(.*?)\}",
        r"answer is (.*?)(\.|$)",
        r"(?:^|\n)(\d+)(?:$|\n)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return None

def normalize_answer(text):
    if text is None:
        return ""
    # Remove all whitespace and convert to lower case for comparison
    return re.sub(r"\s+", "", text).lower()

def main():
    path = os.path.expanduser("/mnt/tidal-alsh-hilab/usr/yeliming/models/Qwen/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if "Qwen" in path:
        stop_token_ids = [151645, 151643]  # <|im_end|>, <|endoftext|>
    else:
        stop_token_ids = [tokenizer.eos_token_id]
        
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, kvcache_block_size=1, attention_backend="flashinfer", gpu_memory_utilization=0.8)

    dataset_path = Path("/root/nano-vllm/aime25/aime2025*.jsonl")
    questions = []
    answers = []

    for file in dataset_path.parent.glob(dataset_path.name):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                questions.append(data["question"])
                answers.append(data["answer"])

    # AIME requires reasoning, so let's allow more output tokens and chain of thought
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=16384,
    )
    
    system_prompt = "You are a helpful assistant. Solve the following math problem step by step and put the final answer in \\(\\mathbf{} \\) when you finish."
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        + "<think>\n"
        for q in questions
    ]
    logger.info("Starting generation for AIME 2025 questions...")
    logger.info(f"prompt[0] is {prompts[0]}")
    logger.info(f"Loaded {len(prompts)} questions from {dataset_path}")
    
    outputs = llm.generate(prompts, sampling_params)

    correct_count = 0
    total_count = len(prompts)
    
    results = []
    
    for i, (prompt, output, true_answer) in enumerate(zip(questions, outputs, answers)):
        generated_text = output['text']
        extracted = extract_answer(generated_text)
        
        is_correct = normalize_answer(extracted) == normalize_answer(true_answer)
        if is_correct:
            correct_count += 1
            
        result_entry = {
            "question_id": i + 1,
            "question": prompt,
            "model_output": generated_text,
            "extracted_answer": extracted,
            "true_answer": true_answer,
            "is_correct": is_correct
        }
        results.append(result_entry)

    score = correct_count / total_count * 100
    
    output_file = "aime25_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"\nResults saved to {output_file}")
    print(f"Final Score: {score:.2f}% ({correct_count}/{total_count})")

if __name__ == "__main__":
    main()
