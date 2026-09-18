# 08 · Scheduler —— 批处理策略与请求生命周期

**文件:** [`nanovllm/engine/scheduler.py`](../../../nanovllm/engine/scheduler.py)

引擎的"政策层":每个 step 挑哪些序列、算多少 token、缓存命中怎么折算、显存不够抢占谁。vLLM 的三大调度特性——**连续批处理**(continuous batching)、**chunked prefill**、**抢占与重算**——在这 93 行里全部可见,是全仓库逻辑密度最高的文件。

## 前置知识

- **Prefill vs Decode 的成本不对称** — prefill 一次算整个 prompt(算力密集、吞吐高),decode 每步每序列只算 1 个 token(访存密集);调度器优先凑大 prefill 批。
- **连续批处理** — 不等一整批跑完:每步动态增删序列,`waiting`/`running` 双队列是其最小实现。
- **Chunked prefill** — 把超长 prompt 切成多个 step 处理,避免一个巨 prompt 独占整步、阻塞 decode。
- **抢占-重算(preempt-by-recompute)** — 显存不足时把运行中序列踢回等待队列、丢弃其 KV,之后**重新 prefill**;有前缀缓存时重算代价大减。
- 模块 [02](02-sequence.md) 的计数器(`num_scheduled_tokens`/`num_cached_tokens`)与模块 [07](07-block-manager.md) 的接口,这里全面消费。

## 运行时位置

`LLMEngine.step()` 每轮调用一次 `schedule()`(`nanovllm/engine/llm_engine.py:50`),拿到 `(序列列表, 是否 prefill)` 后交给 `ModelRunner`,再用 `postprocess` 收尾。它持有唯一一份 `BlockManager`。

## 从哪里开始读

`schedule()`(`nanovllm/engine/scheduler.py:25-73`)——先读 prefill 段(30-55),再读 decode 段(57-73),然后回头看 `preempt` 和 `postprocess`。

## 核心对象与 API

| 对象 | 位置 | 作用 |
|---|---|---|
| `Scheduler.waiting` / `running` | `nanovllm/engine/scheduler.py:16-17` | 双端队列:待处理 / 运行中 |
| `schedule()` | `nanovllm/engine/scheduler.py:25-73` | 核心:prefill 优先,decode 兜底 |
| `preempt()` | `nanovllm/engine/scheduler.py:75-79` | 踢回 waiting、释放 blocks、标记重新 prefill |
| `postprocess()` | `nanovllm/engine/scheduler.py:81-92` | 记账、追加 token、判定结束 |
| `add()` / `is_finished()` | `nanovllm/engine/scheduler.py:19-23` | 引擎入口/出口 |

## 关键流程

### Prefill 段(`:29-55`)

循环取 `waiting` 队首,在两个预算内装批:`max_num_seqs`(序列数)与 `max_num_batched_tokens`(token 数)。三个值得逐行看的点:

1. **前缀缓存折算**(`:35-39`):新序列先 `can_allocate` 探测;`-1`(池不够)直接 `break` 停止装批;命中 `num_cached_blocks` 块则**只调度剩余 token**(`num_tokens - num_cached_blocks*block_size`)——这就是前缀缓存省算力的位置。
2. **chunked prefill 只对队首序列开放**(`:42`):`remaining < num_tokens and scheduled_seqs` 时 break——只有批里第一个序列允许被切块续传;已开块的序列 `block_table` 非空,继续按 `num_tokens - num_cached_tokens` 调度(`:40-41`)。
3. **整 prompt 调度完才转 RUNNING**(`:48-51`):部分调度(被切块)的序列留在 `waiting` 队首,下个 step 接着算。

### Decode 段与抢占(`:57-73`)

prefill 没装到任何序列时才走 decode:每序列 1 token。关键在 `can_append` 失败(需要新 block 而池满)时:

```python
while not self.block_manager.can_append(seq):
    if self.running:
        self.preempt(self.running.pop())   # 踢掉队尾(最新加入的)
    else:
        self.preempt(seq)                  # 池里只剩自己:踢自己,重算
        break
```

`self.running.pop()` 是 **LIFO 抢占**——牺牲最新序列;`preempt`(`:75-79`)释放其全部 blocks、状态回 `WAITING`、`is_prefill=True`,将来重算时前缀缓存会大量命中(它之前的块哈希都登记过)。

### postprocess:一个循环做完四件事(`:81-92`)

对每个调度过的序列:`hash_blocks`(登记新满块)→ 推进 `num_cached_tokens`、清零 `num_scheduled_tokens` → 未调度完整 prompt 则 `continue`(chunk 续传中)→ `append_token` 并按 EOS/`max_tokens` 判定结束、释放 blocks。

## 常见疑问 / 坑

- **`eos` 从哪来?** `Config.eos` 由引擎在 tokenizer 加载后写入(模块 [01](01-config-and-sampling.md));`ignore_eos=True`(bench 用它保证输出长度可控)会跳过 EOS 判定(`:89`)。
- **潜在硬失败路径**:若第一条 prompt 的块数就超过整个 KV 池,`can_allocate` 返回 `-1` 触发 `:37-38` 的 `break`;此时 `running` 也是空的,decode 段不会执行,`:71` 的 `assert scheduled_seqs` 直接炸——没有友好报错。单条超长 prompt(> `num_kvcache_blocks × block_size` 个 token)会触发此路径。(依据代码路径推断,未实际运行验证。)
- **prefill 批内序列数与 token 数是双预算**(`:30-46`),两者任一耗尽即停;这就是 `max_num_batched_tokens` 名字的由来。
- `schedule()` 返回的 `is_prefill` 指示的是**本批性质**而非序列状态——同一 step 里不会混合 prefill 与 decode(vLLM 后来版本支持混合批,这里刻意简化)。

## 验证

```bash
.venv/bin/python -c "
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler

class C: pass
c = C(); c.max_num_seqs = 4; c.max_num_batched_tokens = 16; c.eos = -1
c.kvcache_block_size = 4; c.num_kvcache_blocks = 8
sch = Scheduler(c)
for i in range(3): sch.add(Sequence(list(range(6))))
seqs, is_prefill = sch.schedule()
print('scheduled', len(seqs), 'prefill', is_prefill)
# 输出 'scheduled 2 prefill True':前两条用掉 12/16 token,第三条放不下被留下
"
```
(此桩测试仅演示接口;完整行为以 [README 端到端验证](../README.md#常用工作流)为准。)
