# 02 · Sequence —— 逐请求状态机

**文件:** [`nanovllm/engine/sequence.py`](../../../nanovllm/engine/sequence.py)

每个用户请求对应一个 `Sequence` 对象。它持有不断增长的 token 列表、prompt/completion 分界、生命周期状态,以及支撑 chunked prefill 与前缀缓存记账的计数器。它同时还是**线上传输格式**:TP worker 进程看不到完整对象,只看到它的压缩 pickle。

## 前置知识

- **`Enum` / `itertools.count`** — 状态标志与单调递增 `seq_id`(`nanovllm/engine/sequence.py:8-16`);`seq_id` 的顺序正是 `LLMEngine.generate` 恢复输出顺序的依据。
- **Pickle 的 `__getstate__`/`__setstate__` 协议** — 该类重定义了两者(`nanovllm/engine/sequence.py:72-83`)以压缩跨进程传输量;理解张量并行模式(模块 [09](09-model-runner.md))时必需。
- **block / block-table 词汇** — `block_table` 挂在 Sequence 上,但由 `BlockManager`(模块 [07](07-block-manager.md))负责管理。

## 运行时位置

由 `LLMEngine.add_request` 创建(`nanovllm/engine/llm_engine.py:43-47`),由 `Scheduler` 排队并改写状态(模块 [08](08-scheduler.md)),每一步经 `ModelRunner.call("run", ...)` pickle 传给 TP worker。

## 从哪里开始读

先读 `Sequence.__init__`(`nanovllm/engine/sequence.py:18-31`),再读其下的属性方法,最后读文件末尾的两个 pickle 方法。

## 核心对象与 API

| 成员 | 位置 | 作用 |
|---|---|---|
| `SequenceStatus` | `nanovllm/engine/sequence.py:8-11` | `WAITING → RUNNING → FINISHED` 生命周期 |
| `Sequence.block_size` | `nanovllm/engine/sequence.py:15` | **类**属性;启动时被 `Config` 覆写(`nanovllm/engine/llm_engine.py:21`) |
| `token_ids` / `num_prompt_tokens` | `nanovllm/engine/sequence.py:21-24` | 完整 token 历史;prompt 长度固定了 prompt/completion 分界 |
| `num_cached_tokens` | `nanovllm/engine/sequence.py:25` | 命中前缀缓存、免于重复预填充的 token 数 |
| `num_scheduled_tokens` | `nanovllm/engine/sequence.py:26` | 本 step 被调度的 token 数(仅 chunked prefill 进行中非零) |
| `block_table` | `nanovllm/engine/sequence.py:28` | 存放该序列 KV cache 的 block id 列表 |
| `num_blocks`、`last_block_num_tokens`、`block(i)` | `nanovllm/engine/sequence.py:56-65` | 分块数学:向上取整除法与逐 block 的 token 切片 |
| `append_token` | `nanovllm/engine/sequence.py:67-70` | decode step 唯一的增长操作 |
| `__getstate__` / `__setstate__` | `nanovllm/engine/sequence.py:72-83` | 面向 TP 共享内存通道的压缩 pickle |

## 关键流程

1. **准入。** `add_request` 用 token id 列表构造 `Sequence`:状态 `WAITING`、`is_prefill=True`、`block_table` 为空。
2. **预填充。** Scheduler 设定 `num_scheduled_tokens`(可能小于整个 prompt → chunked prefill);`BlockManager.allocate` 填充 `block_table` 与 `num_cached_tokens`。
3. **解码。** 整个 prompt 调度完后,Scheduler 把状态翻成 `RUNNING`(`nanovllm/engine/scheduler.py:48-51`);此后每步每序列只追加一个 token。
4. **结束。** 命中 EOS 或达到 `max_tokens` → 状态 `FINISHED`,释放 blocks(`nanovllm/engine/scheduler.py:89-92`)。

### pickle 技巧

`__getstate__` 在预填充阶段发送**完整 token 列表**,解码阶段只发送 `last_token`(`nanovllm/engine/sequence.py:72-74`)。decode 时 worker 侧重建出的对象 `token_ids = []`——这没有问题,因为 decode 阶段注意力从分页缓存按 `block_table` 读 K/V,根本不需要 token 历史。一行 pickle 定制,砍掉了 TP 模式大部分每步 IPC 开销。

## 常见疑问 / 坑

- `block_size` 是被引擎以**类属性**方式改写的(`nanovllm/engine/sequence.py:21`)——任何在 `LLMEngine.__init__` 之前构造 `Sequence` 的代码(如 `ModelRunner.warmup_model`,`nanovllm/engine/model_runner.py:97`)用的都是默认值或上一次的值。
- `__setstate__` 重建的是**残缺**对象:`prompt_token_ids` 之类的属性只有在持有完整 `token_ids` 的 rank 0 上才有意义。

## 验证

```bash
.venv/bin/python -c "
from nanovllm.engine.sequence import Sequence
s = Sequence(list(range(300)))
print(s.num_blocks, s.last_block_num_tokens, s.block(1)[:3])"
```
