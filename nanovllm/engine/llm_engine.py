import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 从Config类中获取所有字段名，并创建一个集合
        config_fields = {field.name for field in fields(Config)}
        # 从kwargs中过滤出config_fields中的字段，并创建一个字典
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # **为解包语法，用于将字典解包为关键字参数传递给 Config 的构造函数
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        # Python多进程有三种启动方式：spawn、fork、forkserver
        # spawn：在子进程中启动新的Python解释器，子进程与父进程完全隔离，子进程的运行环境与父进程完全独立
        # fork：在子进程中启动新的Python解释器，子进程与父进程共享相同的内存空间，子进程的运行环境与父进程完全相同
        # forkserver：在子进程中启动新的Python解释器，子进程与父进程共享相同的内存空间，子进程的运行环境与父进程完全相同
        # 这里使用spawn方式，因为spawn方式可以完全隔离子进程与父进程，子进程的运行环境与父进程完全独立，不会有CUDA上下文的问题
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加请求，将提示词编码为 token id，并添加到调度器中

        Args:
            prompt (str | list[int]): 提示词
            sampling_params (SamplingParams): 采样参数
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """
        进行一次 decode，自回归产生一个 token
        """
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """
        输入 prompt 提示词，以及各条 prompt 的采样参数，根据各个 prompt 请求及其采样参数，生成文本

        Args:
            prompts (list[str] | list[list[int]]): 提示词
            sampling_params (SamplingParams | list[SamplingParams]): 采样参数
            use_tqdm (bool, optional): 是否使用tqdm显示进度条

        Returns:
            list[str]: 生成的文本
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
