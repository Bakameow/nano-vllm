# 10 · LLMEngine 与对外门面 —— 组装根与生成主循环

**文件:** [`nanovllm/engine/llm_engine.py`](../../../nanovllm/engine/llm_engine.py) · [`nanovllm/llm.py`](../../../nanovllm/llm.py) · [`nanovllm/__init__.py`](../../../nanovllm/__init__.py) · 入口:[`example.py`](../../../example.py) · [`bench.py`](../../../bench.py)

组合根:把 Config、TP 进程、ModelRunner、Scheduler、tokenizer 组装起来,并提供 `generate()` 主循环。到此为止,前九个模块的全部零件在这里拧成一台发动机——**这也是适合最后读、却最适合作为断点调查入口的文件**。

## 前置知识

- **`multiprocessing` spawn 上下文** — CUDA 不支持 fork;`mp.get_context("spawn")` 子进程从零重新 import 并 pickle 传参。
- **前 9 个模块的接口** — 尤其 `Scheduler.schedule/postprocess`(模块 [08](08-scheduler.md))与 `ModelRunner.call`(模块 [09](09-model-runner.md));本模块几乎不含新逻辑,价值在于组装顺序。

## 运行时位置

用户直接接触的唯一入口:`from nanovllm import LLM`。`LLM` 只是 `LLMEngine` 的空子类(`nanovllm/llm.py:4-5`),沿用 vLLM 的 API 形状。

## 从哪里开始读

从 `generate()`(`nanovllm/engine/llm_engine.py:60-90`)倒推:它调 `add_request` 与 `step`,`step` 调 `schedule → model_runner.call → postprocess`;最后回头读 `__init__` 看零件如何造出来。

## 核心对象与 API

| 对象 | 位置 | 作用 |
|---|---|---|
| `LLMEngine.__init__` | `nanovllm/engine/llm_engine.py:17-35` | 组装根:Config → TP 进程 → ModelRunner → tokenizer → Scheduler |
| `LLMEngine.step` | `nanovllm/engine/llm_engine.py:49-55` | 一轮推理:schedule → run → postprocess |
| `LLMEngine.generate` | `nanovllm/engine/llm_engine.py:60-90` | 主循环 + tqdm 吞吐显示 + 输出还原 |
| `LLM` | `nanovllm/llm.py:4-5` | 公开门面,纯继承 |
| `example.py` / `bench.py` | 仓库根 | 两个可运行入口 |

## 关键流程

### 组装顺序里的三个细节(`:17-35`)

1. **kwargs → Config 过滤**(`:18-19`):未知名静默丢弃(模块 [01](01-config-and-sampling.md) 的坑)。
2. **TP 进程模型**(`:22-31`):rank 1..N-1 逐个 `spawn`,各带一个 `Event`;rank 0 在**本进程内**构造 `ModelRunner(config, 0, self.events)`——注意它拿到的是**全部 Event 列表**(广播用),worker 只拿自己的一个。结合模块 [09](09-model-runner.md):worker 的 `__init__` 永不返回。
3. **两次跨模块副作用**(`:21,33`):`Sequence.block_size`(类属性)与 `Config.eos`(延迟字段)都在这里补上;`atexit.register(self.exit)`(`:35`)保证退出时向 worker 广播 `"exit"` 并 `join`。

### step() 与 generate():引擎的最小循环(`:49-90`)

```python
seqs, is_prefill = self.scheduler.schedule()          # 政策(模块 08)
token_ids = self.model_runner.call("run", seqs, is_prefill)   # 执行(模块 09)
self.scheduler.postprocess(seqs, token_ids, is_prefill)       # 记账(模块 08)
```

`generate()` 在此之上做四件事:请求入队;循环 `step` 直到 `is_finished()`;用 `num_tokens` 的正负区分 prefill/decode 吞吐并刷上 tqdm(`:75-83`);结束后按 `seq_id` 排序还原请求顺序、detokenize(`:88-89`)。返回 `{"text": ..., "token_ids": ...}` 列表。

### 两个入口脚本

- [`example.py`](../../../example.py):chat template 包 prompt → `generate`,是**最小可读示例**。
- [`bench.py`](../../../bench.py):随机 token id 批量压测;先 `llm.generate(["Benchmark: "], ...)` **暖机**(`bench.py:22`,预热 compile/CUDA graph/flash-attn)再计时,`ignore_eos=True` 保证输出长度可控。

## 常见疑问 / 坑

- **`sampling_params` 传单对象即可**:内部自动复制 N 份对齐 prompts(`:67-68`)。
- **输出顺序 = 输入顺序**,靠 `sorted(outputs.keys())` 而非完成顺序(`:88`)。
- **没有任何服务化组件**:没有 HTTP server、没有异步 API——这是纯离线批量推理引擎;`exit()` 后进程组销毁,`LLM` 对象基本不可复用重建(TP 模式下共享内存名固定,见模块 [09](09-model-runner.md))。

## 验证

端到端验证就是跑 `example.py`/`bench.py`(见 [README 常用工作流](../README.md#常用工作流)),本机已用真实权重跑通,记录见 [端到端验证记录](../README.md#端到端验证记录)。
