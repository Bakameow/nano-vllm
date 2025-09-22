import datasets
import json
import re
import os
import random
from transformers import AutoTokenizer
from pathlib import Path
from nanovllm import LLM, SamplingParams
from nanovllm.utils.log import log_set_level
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--file', '-f', action='store', default='aime24_answers.json')
parser.add_argument('--check', '-c', action='store_true')
parser.add_argument('--test', '-t', action='store_true')
parser.add_argument('--batch_size', '-b', action='store', type=int, default=5)
parser.add_argument('--log_level', '-l', action='store', type=str, default='DEBUG')
parser.add_argument('--model', '-m', action='store', type=str, default='Qwen3-8B')
random.seed(42)

def format_options(options):
    letters = "ABCDEFGHIJ"
    lines = [f"({letters[i]}): {opt}" for i, opt in enumerate(options)]
    return "Options:\n" + "\n".join(lines)

def get_prediction(output: str, pattern: str, mismatch_count: int) -> tuple[str, int]:
    match = re.search(pattern, output)
    if match:
        return match.group(0), mismatch_count
    else:
        return "Unknown", mismatch_count+1

def format_answers():
    answers = []
    with open(answers_path, 'r+') as f:
        for line in f:
            entry = json.loads(line)
            entry['Solution'] = entry['Solution']
            del entry['solution']
            answers.append(entry)

    with open(answers_path.with_suffix('.jsonl'), 'w') as f:
        for entry in answers:
            f.write(json.dumps(entry) + '\n')

def check_accuracy(answers_path: Path):
    test_df = read_dataset()
    
    success, fail = 0, 0
    mismatch = 0
    with open(answers_path, 'r') as f:
        for i, line in tqdm(enumerate(f), desc="Checking accuracy"):
            entry = json.loads(line)
            prediction, mismatch = get_prediction(entry['Solution'], str(entry['answer']), mismatch)
            # print(prediction, str(entry['Answer']))
            if prediction == str(entry['answer']):
                success += 1
            else:
                fail += 1
    print("-"*100)
    print('length of test_df: ', len(test_df))
    print('length of jsonl: ', success + fail)
    print('mismatch: ', mismatch)
    print('average accuracy: ', success / (success + fail))
    print("-"*100)


def read_dataset():
    data_files = {
        'train': '/sgl-workspace/nano-vllm/tmp/aime_2025/*.jsonl',
    }
    dataset : datasets.DatasetDict = datasets.load_dataset("json", data_files=data_files)
    dataset['train'] = dataset['train'].add_column("ID", range(len(dataset['train'])))
    if args.test:
        print(dataset['train'].features)
        keys = dataset['train'].features.keys()
        for item in dataset['train']:
            for k in keys:
                print(k, ":", item[k])
    return dataset['train']


def main():
    test_df = read_dataset()

    success, fail = 0, 0

    finished_question_ids = []
    if not answers_path.exists():
        answers_path.touch()
    with open(answers_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            finished_question_ids.append(tmp['ID'])
    test_df = test_df.filter(lambda x: x['ID'] not in finished_question_ids)
    print('finished_question:',len(finished_question_ids))
    print('unfinished_question:',len(test_df))
    batch_size = args.batch_size
    num_batches = len(test_df) // batch_size

    path = Path("~/.cache/modelscope/hub/models/Qwen/" + args.model).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(path)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    print('----------------- Start Answering -------------------')
    for start in tqdm(range(0, len(test_df), batch_size),
                  total=num_batches, desc="Inference", unit="batch"):
        batch = test_df.select(range(start, min(start + batch_size, len(test_df))))
        querys = []
        answers = []
        for item in batch:
            query = 'Q: ' + item['question'] + '\n'
            querys.append(query)
        if len(querys) == 0:
            continue

        prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for query in querys
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        # 对于已完成的请求（已从 batch 中移除），不再重复提问
        mismatch = 0
        for id, item in enumerate(batch):
            item['Solution'] = outputs[id]['text']
            answers.append(item)
            prediction, mismatch = get_prediction(outputs[id]['text'], str(item['answer']), mismatch)
            if str(item['answer']) == prediction:
                success += 1
            else:
                fail += 1


        with open(answers_path, 'a') as f:
            for item in answers:
                f.write(json.dumps(item) + '\n')

def test():
    path = Path("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B").expanduser()
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    list_of_numbers = [str(x) for x in range(10)]
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
        print(f"Prompt: {prompt!r}")
        print(f"Output: {output['text']}")
        print(f"Completion: {len(tokenizer.encode(prompt)) + len(output['token_ids'])}")
    
if __name__ == "__main__":
    
    args = parser.parse_args()
    answers_path = Path(args.file)
    log_set_level(args.log_level)
    if args.check:
        check_accuracy(answers_path)
    elif args.test:
        read_dataset()
        test()
    else:
        main()

# python aime25.py -f results/qwen3_8B_aime25.jsonl -c
# python aime25.py -f results/qwen3_0_6B_aime25.jsonl -m Qwen3-0___6B
# python aime25.py -t
# python aime25.py -f results/qwen3_0_6B_aime25.jsonl -c