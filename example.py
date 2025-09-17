import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer
from pathlib import Path

def main():
    path = Path("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B").expanduser()
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024)
    list_of_numbers = [str(x) for x in range(1000)]
    prompts = [
        " ".join(list_of_numbers),
        "list all prime numbers within 1000, just list the numbers",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    print(prompts)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {len(tokenizer.encode(prompt)) + len(output['token_ids'])}")


if __name__ == "__main__":
    main()
