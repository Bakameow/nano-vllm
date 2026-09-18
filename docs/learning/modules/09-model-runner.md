# 09 · ModelRunner —— GPU 执行器、TP 控制面与 CUDA graph

**文件:** [`nanovllm/engine/model_runner.py`](../../../nanovllm/engine/model_runner.py)

全仓库工程含量最高的文件:进程组初始化、显存预算与 KV 池分配、CUDA graph 捕获与重放、每步输入张量构造,以及一套基于共享内存 + 事件的自制 TP 控制面。Scheduler 决定"做什么",这里决定"怎么在 N 张卡上跑起来"。

## 前置知识

- **`torch.distributed` NCCL 进程组** — `init_process_group("nccl", ...)` + `torch.cuda.set_device(rank)`;TP 各 rank 执行相同计算,靠集合通信保持一致(模块 [05](05-tensor-parallel-layers.md))。
- **CUDA graph** — 把一串 kernel 录制成可单次重放的图;重放要求**输入缓冲地址固定**,所以有 `graph_vars` 静态缓冲,更新数据而非换张量。
- **flash-attn varlen 元数据** — `cu_seqlens_q/k`(各序列边界的累计和)、`max_seqlen_q/k`、`slot_mapping`(token → KV 缓存槽位)、`block_tables`(模块 [04](04-attention-and-context.md) 消费它们)。
- **多进程 `SharedMemory` / `Event`** — rank 0 广播"调用什么方法"给 worker 的轻量 RPC。
- **CUDA 显存统计 API** — `mem_get_info` / `memory_stats()["...peak"]` 是 KV 池容量公式的输入。

## 运行时位置

rank 0 的 `ModelRunner` 活在引擎进程里(`nanovllm/engine/llm_engine.py:31`),rank 1..N-1 是 `multiprocessing.spawn` 子进程(`:25-30`)。每 step 被 `LLMEngine.step()` 调一次 `call("run", seqs, is_prefill)`。

## 从哪里开始读

先读 `run()`(`nanovllm/engine/model_runner.py:214-220`)把整个 step 串起来,再按 `prepare_prefill` → `prepare_decode` → `run_model` → `allocate_kv_cache` → `capture_cudagraph` → `call/read_shm/write_shm` 的顺序展开。

## 核心对象与 API

| 对象 | 位置 | 作用 |
|---|---|---|
| `ModelRunner.__init__` | `nanovllm/engine/model_runner.py:17-48` | 初始化全程:进程组 → 建模 → 加载 → 预热 → KV 池 → CUDA graph |
| `allocate_kv_cache` | `nanovllm/engine/model_runner.py:103-121` | 显存预算公式 + 缓存视图注入 Attention 层 |
| `warmup_model` | `nanovllm/engine/model_runner.py:91-101` | 最大形状空跑,测显存峰值(为 KV 池让路) |
| `prepare_prefill` | `nanovllm/engine/model_runner.py:129-170` | 打包 input_ids/positions/cu_seqlens/slot_mapping |
| `prepare_decode` | `nanovllm/engine/model_runner.py:172-188` | 每序列取 `last_token`,槽位 = 末块下一格 |
| `run_model` | `nanovllm/engine/model_runner.py:195-212` | eager 或 CUDA graph 重放二选一 |
| `capture_cudagraph` | `nanovllm/engine/model_runner.py:222-257` | 按 batch 桶捕获 + 共享内存池 |
| `call` / `write_shm` / `read_shm` / `loop` | `nanovllm/engine/model_runner.py:61-89` | TP 控制面:rank 0 广播方法调用 |

## 关键流程

### 启动序列(`:17-48`)——顺序即依赖

```
init_process_group(nccl)          # 必须最先:layers 里 dist.get_rank() 依赖它
→ set_default_device/dtype(cuda)  # 之后建的所有参数直接落在 GPU、模型 dtype
→ Qwen3ForCausalLM + load_model   # 模块 06
→ warmup_model()                  # 空跑测显存峰值
→ allocate_kv_cache()             # 按峰值算 KV 池大小
→ capture_cudagraph()             # 除非 enforce_eager
→ (rank>0) loop() 永驻           # worker 进程 __init__ 永不返回!
```

注意最后一行:worker 进程的 `ModelRunner.__init__` **永远不会返回**——它停在 `self.loop()`(`:48`)里等 rank 0 的调用。理解这点才能读懂 `LLMEngine` 的进程模型。

### KV 池容量公式(`:103-121`)

```python
num_blocks = (total × gpu_memory_utilization − used − peak + current) // block_bytes
```

`used` 是进程外已占显存,`peak − current` 是预热时分配又释放的暂存(权重加载、编译缓存),真正留给 KV 池。`kv_cache` 形状 `(2, 层数, blocks, block_size, kv_heads, head_dim)`(`:115`),随后**按模块遍历**,把每层的 `k_cache/v_cache` 指向对应切片(`:117-121`)——这是纯运行时注入,import 图上看不到这条边。

### prepare_prefill:slot_mapping 的算术(`:129-170`)

对每个序列,从 `num_cached_tokens`(前缀缓存命中数)开始,把待算 token 映射到缓存槽位:逐块算 `block_table[i]*block_size` 加块内偏移(`:151-161`)。**只有存在前缀命中时才构造 `block_tables`**(`:162-163`,判据 `cu_seqlens_k > cu_seqlens_q`)——常规 prefill 直接吃新算的 k/v,不需要分页寻址(呼应模块 [04](04-attention-and-context.md))。

### CUDA graph:桶 + 静态缓冲(`:195-212, 222-257`)

- 捕获的 batch 桶:`[1, 2, 4, 8] + range(16, max_bs+1, 16)`(`:234`);**只捕 decode**。
- `graph_vars` 是预分配的静态张量字典(`:250-257`);重放前把本批数据**拷进**缓冲、`fill_(-1)` 清空 slot_mapping、`graph.replay()`(`:200-212`)。多余行的 slot 为 `-1`,Triton kernel 跳过(模块 [04](04-attention-and-context.md))。
- 触发条件(`:197`):非 prefill、非 eager、且 `bs ≤ 512` 才走 graph;否则 eager。

### TP 控制面:`call` 的双跳(`:61-89`)

rank 0 的 `call(method, *args)` 做两件事:先把 `[method, args]` pickle 进 1MB 共享内存并 `set` 所有 worker 的 `Event`(`write_shm`),再**自己本地执行同一方法**。worker 从 `loop()` 里醒来、反序列化、执行——控制流对齐了,数值一致性交给集合通信(all_reduce/gather)。返回值只从 rank 0 取(`run` 里 `self.rank == 0` 才采样,`:216-218`)。

## 常见疑问 / 坑

- **固定端口 `tcp://localhost:2333`**(`:26`):并行跑两个实例会端口冲突直接崩。
- **共享内存名为固定的 `"nanovllm"`**(`:43`):同机两个 TP 实例同样冲突。
- `warmup_model` 用的 `Sequence` 是绕过引擎构造的(`:97`),`block_table` 为空,`prepare_prefill` 里有专门分支跳过 slot 映射(`:149-150`)。
- 捕获阶段每个桶先 warmup 再 capture(`:240-243`),首次切换 batch 桶大小的那步 decode 会略慢,属正常。

## 验证

单卡 + 本机权重跑最小推理(覆盖 init/warmup/KV 分配/eager 路径):

```bash
MODEL=/home/tiger/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
.venv/bin/python -c "
from nanovllm import LLM, SamplingParams
llm = LLM('$MODEL', enforce_eager=True)
out = llm.generate(['The capital of France is'], SamplingParams(temperature=0.6, max_tokens=8))
print(out[0]['text'])"
```
(CUDA graph 路径把 `enforce_eager` 去掉即可;结果见 [README](../README.md#端到端验证记录)。)
