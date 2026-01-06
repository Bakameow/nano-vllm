import os
import json
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from loguru import logger

def extract_answer(text):
    # Try different patterns to extract the answer
    patterns = [
        r"####\s+(.*?)(\n|$)",  # Standard GSM8K format
        r"\\mathbf\{(.*?)\}",
        r"boxed\{(.*?)\}",
        r"answer is (.*?)(\.|$)",
        r"(?:^|\n)(\d+)(?:$|\n)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # Remove commas from numbers (e.g., 1,000 -> 1000)
            return match.group(1).strip().replace(',', '')
    return None

def normalize_answer(text):
    if text is None:
        return ""
    # Remove all whitespace and convert to lower case for comparison
    # Also remove trailing .0 for numbers
    text = re.sub(r"\s+", "", text).lower()
    if text.endswith(".0"):
        text = text[:-2]
    return text

def main():
    path = os.path.expanduser("/mnt/tidal-alsh-hilab/usr/yeliming/models/Qwen/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if "Qwen" in path:
        stop_token_ids = [151645, 151643]  # <|im_end|>, <|endoftext|>
    else:
        stop_token_ids = [tokenizer.eos_token_id]
        
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, kvcache_block_size=1, attention_backend="flashinfer", gpu_memory_utilization=0.7)

    dataset_path = Path("data/gsm8k/main/test-00000-of-00001.parquet")
    logger.info(f"Loading dataset from {dataset_path}")
    
    df = pd.read_parquet(dataset_path)
    questions = df["question"].tolist()
    # GSM8K answers usually contain reasoning followed by #### answer
    # We need to extract the final numerical answer from the reference
    raw_answers = df["answer"].tolist()
    answers = []
    
    for ans in raw_answers:
        extracted = extract_answer(ans)
        if extracted:
             answers.append(extracted)
        else:
             # Fallback: take the last number or the whole string if extraction fails
             # GSM8K usually has #### 
             parts = ans.split("#### ")
             if len(parts) > 1:
                 answers.append(parts[-1].strip().replace(',', ''))
             else:
                 answers.append(ans.strip())

    # GSM8K requires reasoning
    sampling_params = SamplingParams(
        temperature=0.0, 
        max_tokens=2048,
    )
    
    # Updated system prompt for GSM8K
    system_prompt = "You are a helpful assistant. Solve the following math problem step by step and put the final answer after ####."
    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        # + "<think>\n" # Removed specific R1/DeepSeek thinking tag for general compatibility, add back if model is specifically trained for it. 
        # But user requested apply_chat_template with output format matching, keeping <think> if using reasoning model
        + "<think>\n" # Kept for consistency with previous file edits if using similar model
        for q in questions
    ]
    
    # Limit for testing if needed, though running full test set is standard
    # questions = questions[:10]
    # answers = answers[:10]
    # prompts = prompts[:10]
    
    logger.info("Starting generation for GSM8K questions...")
    logger.info(f"prompt[0] is {prompts[0]}")
    logger.info(f"Loaded {len(prompts)} questions")
    
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
    
    output_file = "gsm8k_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"\nResults saved to {output_file}")
    print(f"Final Score: {score:.2f}% ({correct_count}/{total_count})")

if __name__ == "__main__":
    main()
