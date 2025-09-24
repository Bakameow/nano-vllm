from nanovllm import LLM, SamplingParams
from nanovllm.utils.log import log_set_level
from transformers import AutoTokenizer
from pathlib import Path

def main():
    path = Path("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B").expanduser()
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, attention_backend="flex_attention", max_num_seqs=5)
    # llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, attention_backend="flash_attention", max_num_seqs=5)
    log_set_level("INFO")
    sampling_params = SamplingParams(temperature=0, max_tokens=100)
    list_of_numbers = [str(x) for x in range(1000)]
    # print(list_of_numbers)
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
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        # print(f"Prompt: {prompt!r}")
        print(f"{output['text']}")
        print(f"Completion: {len(tokenizer.encode(prompt)) + len(output['token_ids'])}")


if __name__ == "__main__":
    main()
