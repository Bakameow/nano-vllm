
import datasets
import json
import re
import os
import random
from transformers import AutoTokenizer
from pathlib import Path
from nanovllm import LLM, SamplingParams
from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--file', '-f', action='store', default='mmlu_pro_answers.json')
parser.add_argument('--check', '-c', action='store_true')
random.seed(42)

def format_options(options):
    letters = "ABCDEFGHIJ"
    lines = [f"({letters[i]}): {opt}" for i, opt in enumerate(options)]
    return "Options:\n" + "\n".join(lines)

def get_prediction(output: str):
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, output)
    if match:
        return match.group(1)
    else:
        # print(output)
        print("extraction failed, do a random guess")
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

def format_answers():
    answers = []
    with open(answers_path, 'r+') as f:
        for line in f:
            entry = json.loads(line)
            entry['solution'] = entry['solution']["text"]
            answers.append(entry)

    with open(answers_path.with_suffix('.jsonl'), 'w') as f:
        for entry in answers:
            f.write(json.dumps(entry) + '\n')

def check_accuracy(answers_path: Path):
    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                'psychology', 'history']
    per_category_accuracy = {c: [0, 0] for c in categories}
    success, fail = 0, 0
    with open(answers_path, 'r') as f:
        for i, line in tqdm(enumerate(f), desc="Checking accuracy"):
            entry = json.loads(line)
            prediction = get_prediction(entry['solution'])
            if prediction == entry['answer']:
                success += 1
                per_category_accuracy[entry['category']][0] += 1
            else:
                fail += 1
                per_category_accuracy[entry['category']][1] += 1
    
    for k, v in per_category_accuracy.items():
        print('accuracy: ', k, v[0] / (v[0] + v[1]))
    print('average accuracy: ', success / (success + fail))


def read_dataset():
    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                'psychology', 'history']
    sample_per_label = 20
    dataset = datasets.load_dataset("/root/.cache/modelscope/hub/datasets/modelscope/MMLU-Pro/data")
    test_df, val_df = dataset["test"], dataset["validation"]

    filtered_subset = []
    for category in categories:
        category_df = test_df.filter(lambda x: x['category'] == category)
        offset = min(len(category_df), sample_per_label)
        category_df = category_df.select(range(offset))
        filtered_subset.append(category_df)
    
    final_df = datasets.concatenate_datasets(filtered_subset)
    return final_df


def main():
    dataset = datasets.load_dataset("/root/.cache/modelscope/hub/datasets/modelscope/MMLU-Pro/data")

    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                  'psychology', 'history']

    # load 5-shot prompts for each category
    prompts = {c: '' for c in categories}
    for d in dataset['validation']:
        prompts[d['category']] += 'Q:' + ' ' + d['question'] + '\n' + format_options(d['options']) + '\n' + d['cot_content'] + '\n\n'

    per_category_accuracy = {c: [0, 0] for c in categories}
    success, fail = 0, 0

    answers = []
    finished_question_ids = []
    if not answers_path.exists():
        answers_path.touch()
    with open(answers_path, 'r') as f:
        for line in f:
            tmp = json.loads(line)
            finished_question_ids.append(tmp['question_id'])

    test_ds = read_dataset()
    batch_size = 2
    num_batches = len(test_ds) // batch_size

    path = Path("~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B").expanduser()
    tokenizer = AutoTokenizer.from_pretrained(path)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=4096)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    print('----------------- Start Answering -------------------')

    for start in tqdm(range(0, len(test_ds), batch_size),
                  total=num_batches, desc="Batches", unit="batch"):
        batch = test_ds.select(range(start, min(start + batch_size, len(test_ds))))
        querys = []
        for entry in batch:
            # print(entry)
            if entry['question_id'] in finished_question_ids:
                continue
            prefix = prompts[entry['category']]
            query = prefix + 'Q: ' + entry['question'] + '\n' + format_options(entry['options']) + '\n'
            querys.append(query)
        if len(querys) == 0:
            continue
        querys = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for query in querys
        ]
        outputs = llm.generate(querys, sampling_params, use_tqdm=False)
        # print(outputs)
        for id, entry in enumerate(batch):
            entry['solution'] = outputs[id]['text']
            answers.append(entry)
            prediction = get_prediction(outputs[id]['text'])
            if entry["answer"] == prediction:
                success += 1
                per_category_accuracy[entry['category']][0] += 1
            else:
                fail += 1
                per_category_accuracy[entry['category']][1] += 1


        with open(answers_path, 'a') as f:
            for entry in answers:
                f.write(json.dumps(entry) + '\n')

    for k, v in per_category_accuracy.items():
        print('accuracy: ', k, v[0] / (v[0] + v[1]))

if __name__ == "__main__":
    args = parser.parse_args()
    answers_path = Path(args.file)
    if args.check:
        check_accuracy(answers_path)
    else:
        main()

# python mmlu.py -f qwen3_0_6B_mmlu_pro.jsonl -c
# python mmlu.py -f drop_qwen3_0_6B_mmlu_pro.jsonl