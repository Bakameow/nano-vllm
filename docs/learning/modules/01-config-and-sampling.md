# 01 · 配置与采样参数

**文件:** [`nanovllm/config.py`](../../../nanovllm/config.py) · [`nanovllm/sampling_params.py`](../../../nanovllm/sampling_params.py)

两个零内部依赖的小型 dataclass。`Config` 承载启动时一次性确定的所有引擎级参数;`SamplingParams` 承载用户每次调用 `generate()` 时传入的逐请求解码参数。引擎其余所有模块都从这里读取配置——所以它排在第一位。

## 前置知识

- **Python `dataclass(slots=True)`** — 两个类型都是普通 dataclass,校验逻辑全在 `__post_init__`,没有任何配置框架。
- **Hugging Face `AutoConfig`** — `Config.__post_init__` 会加载权重目录下的 `config.json`(`nanovllm/config.py:24`),层数、头数、dtype 等架构事实来自模型本身,而不是用户。
- **分页 KV 词汇(block、block table)** — `kvcache_block_size` / `num_kvcache_blocks` 要到模块 [07](07-block-manager.md) 才有完整含义,现在只需记住它们存在。

## 运行时位置

`Config` 在 `LLMEngine.__init__` 中最先被构造(`nanovllm/engine/llm_engine.py:20`),随后被 pickle 进每个张量并行 worker 进程。`SamplingParams` 由用户构造,随请求进入 `Sequence`(模块 [02](02-sequence.md))。

## 从哪里开始读

两个文件都不到 30 行,从头读到尾即可。

## 核心对象与 API

| 对象 | 位置 | 作用 |
|---|---|---|
| `Config` | `nanovllm/config.py:6` | 引擎级配置 + 缓存 `hf_config` |
| `Config.max_num_batched_tokens` | `nanovllm/config.py:9` | 每个 scheduler step 的 token 预算(默认 16384) |
| `Config.max_num_seqs` | `nanovllm/config.py:10` | 每个 batch 的最大序列数(默认 512) |
| `Config.kvcache_block_size` | `nanovllm/config.py:17` | 每个分页 KV block 存多少 token(默认 256,必须是 256 的倍数) |
| `Config.num_kvcache_blocks` | `nanovllm/config.py:18` | KV 池大小(以 block 计)——`-1` 表示"稍后计算" |
| `Config.enforce_eager` | `nanovllm/config.py:14` | 关闭 CUDA graph 捕获 |
| `SamplingParams` | `nanovllm/sampling_params.py:4` | 逐请求参数:`temperature`、`max_tokens`、`ignore_eos` |

## 关键流程

1. `LLM(...)` 的 kwargs 按已知 `Config` 字段名过滤(`nanovllm/engine/llm_engine.py:18-19`)后转发。
2. `Config.__post_init__` 断言模型路径存在、block size 合法、TP 数 1–8,加载 `hf_config`,并把 `max_model_len` 钳制到模型的 `max_position_embeddings`(`nanovllm/config.py:20-25`)。
3. 有两个字段是**延迟填充**的,由后续阶段写入:
   - `Config.eos`(默认 `-1`)→ 在 `nanovllm/engine/llm_engine.py:33` 由 tokenizer 补上。
   - `Config.num_kvcache_blocks`(默认 `-1`)→ 在 `nanovllm/engine/model_runner.py:113` 按空闲显存算出。
4. 请求到达时,`Sequence.__init__` 只拷贝它需要的三个 `SamplingParams` 字段(`nanovllm/engine/sequence.py:29-31`)。

## 常见疑问 / 坑

- **kwargs 静默过滤。** 拼错选项(如 `tensor_paralel_size=2`)会被静默忽略,不会报错(`nanovllm/engine/llm_engine.py:19`)。
- **不支持贪心解码。** `SamplingParams.__post_init__` 断言 `temperature > 1e-10`(`nanovllm/sampling_params.py:11`),采样器(模块 [03](03-compiled-kernels.md))没有 argmax-only 路径。
- **`eos` 的时序。** TP worker 进程拿到的 `Config` 是 `eos` 填充**之前**的 pickle 快照——目前无害,因为只有 rank 0 的 scheduler 读 `eos`;但若新增 worker 侧的 config 消费者,要小心这一点。

## 验证

```bash
.venv/bin/python -c "
from nanovllm.sampling_params import SamplingParams
print(SamplingParams(temperature=0.6, max_tokens=32))"
```

构造 `Config` 需要真实模型目录(内部有 `os.path.isdir` 断言),所以要等权重就位后才会被真正执行(见 [常用工作流](../README.md#常用工作流))。本机权重位于
`/home/tiger/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master`。
