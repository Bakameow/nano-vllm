# 07 · BlockManager —— 分页 KV 池与前缀缓存

**文件:** [`nanovllm/engine/block_manager.py`](../../../nanovllm/engine/block_manager.py)

vLLM 三大核心机制的其中两个在这里落地:**分页 KV 管理**(PagedAttention 的数据结构侧)与**前缀缓存**(prefix caching)。注意它只做**逻辑记账**——block 里那些 `token_ids` 与 `hash` 是索引用的元数据,真正的 K/V 张量在 GPU 上,由模块 [09](09-model-runner.md) 分配并把视图注入 Attention 层(模块 [04](04-attention-and-context.md))。

## 前置知识

- **分页 KV cache** — 每条序列的 KV 存在若干定长 block 里,序列只持有一张 block id 表(`Sequence.block_table`);物理上不要求连续。
- **引用计数与共享** — 多条序列可共享同一 block(前缀相同时),`ref_count` 归零才真正回收;这是"写时复制"思想的前半截(本仓库没有 COW,共享只读,见下文)。
- **链式哈希 / Merkle 思想** — 每个前缀 block 的哈希 = hash(本块 token + 前一前缀的哈希),使哈希对"整段前缀"敏感,而非单个 block。

## 运行时位置

Scheduler(模块 [08](08-scheduler.md))是它唯一的调用方:`can_allocate/allocate` 在 prefill 准入时,`can_append/may_append` 在 decode 每步,`hash_blocks` 在 `postprocess` 里,`deallocate` 在结束/抢占时。

## 从哪里开始读

`Block` 类(15 行)→ `compute_hash` → `can_allocate`/`allocate` → `hash_blocks`。

## 核心对象与 API

| 对象 | 位置 | 作用 |
|---|---|---|
| `Block` | `nanovllm/engine/block_manager.py:8-23` | `block_id` + `ref_count` + `hash` + `token_ids`(元数据副本) |
| `BlockManager.blocks` | `nanovllm/engine/block_manager.py:30` | 预分配的 block 池(固定长度 `num_blocks`) |
| `free_block_ids` / `used_block_ids` | `nanovllm/engine/block_manager.py:32-33` | 空闲 deque 与在用 set |
| `hash_to_block_id` | `nanovllm/engine/block_manager.py:31` | **前缀缓存索引**:前缀哈希 → block id |
| `compute_hash` | `nanovllm/engine/block_manager.py:35-41` | xxhash(xxh64)链式哈希 |
| `can_allocate` | `nanovllm/engine/block_manager.py:58-73` | 探测最长可复用前缀;容量不够返回 `-1` |
| `allocate` | `nanovllm/engine/block_manager.py:75-92` | 共享缓存块 + 新分配,写 `block_table` 与 `num_cached_tokens` |
| `can_append` / `may_append` | `nanovllm/engine/block_manager.py:103-108` | decode 是否需要新 block;需要则分配 |
| `hash_blocks` | `nanovllm/engine/block_manager.py:110-120` | 把**新写满的** block 登记进前缀缓存索引 |
| `deallocate` | `nanovllm/engine/block_manager.py:94-101` | 引用计数递减,归零回收 |

## 关键流程

### 链式哈希:前缀缓存的正确性来源

`compute_hash(token_ids, prefix_hash)`(`:35-41`)把**上一前缀的哈希**混进本块哈希。于是"相同的前 3 块"必然算出相同的 3 个哈希,而"前两块相同、第三块不同"的序列第三块哈希必然不同——这正是前缀匹配需要的性质。匹配时还会二次校验 `token_ids` 是否真相等(`:66`,防 xxh64 碰撞)。

### 一次 prefill 准入的完整走线

1. `can_allocate(seq)`(`:58-73`):逐块算哈希,在 `hash_to_block_id` 里找最长命中前缀 `num_cached_blocks`;`num_new_blocks` 只把**未被占用的**缓存块计入需要新分配的量。空闲块不足 → 返回 `-1`,由 Scheduler 决定等待或抢占(模块 [08](08-scheduler.md))。
2. `allocate(seq, num_cached_blocks)`(`:75-92`):
   - 命中的块若正被他人使用 → `ref_count += 1` **共享**;若空闲但带哈希 → 直接复用(`ref_count = 1` 并从 free 列表移回,`:86-88`);
   - 其余块走 `_allocate_block` 全新分配;
   - 写 `seq.block_table`,并设 `seq.num_cached_tokens = num_cached_blocks * block_size`——这两行就是前缀缓存全部"收益记账"。
3. 推理后 `hash_blocks(seq)`(`:110-120`):只对**本次新写满**的 block(`start..end`)计算并登记哈希;没写满的尾块不算——因为它的 token 还会增长。

### decode 侧:block 何时增长

`can_append` 的判断只有一行(`:103-104`):`len(seq) % block_size == 1`——即将写入**新块的第一个 token** 时才需要一个新块。`may_append` 据此分配。回收时 `deallocate` **逆序**释放(`:95`)——通常只有尾部若干块是本序列独有的,前缀共享块只减引用计数。

## 常见疑问 / 坑

- **共享不等于 COW**:本仓库前缀命中后直接共享该块的 KV(只读),分叉处之后用新块。没有"复制再改写"的路径,简化了实现。
- **释放 ≠ 忘记**:序列结束后 block 回到 free 队列,但 `hash` 与 `token_ids` 元数据仍在,同前缀的新请求可以认领(`allocate` 的 `:86-88` 分支);块被全新分配时(`_allocate_block`,`:43-51`)旧哈希索引才被删除。
- **块大小必须整除关系成立**:`num_cached_tokens` 等记账全部按"整块"进行,`kvcache_block_size % 256 == 0` 的断言(模块 [01](01-config-and-sampling.md))保证了对齐。
- 哈希用 xxh64(**非加密哈希**),靠 `:66` 的 token 比对兜底防碰撞。

## 验证

BlockManager 是纯 CPU 逻辑,可直接单测式验证:

```bash
.venv/bin/python -c "
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.block_manager import BlockManager
Sequence.block_size = 4                    # 引擎里由 LLMEngine.__init__ 设置(见模块 02 的坑)
bm = BlockManager(num_blocks=8, block_size=4)
s1 = Sequence(list(range(8)))
assert bm.can_allocate(s1) == 0
bm.allocate(s1, 0)
s1.num_scheduled_tokens = 8                # 真实流程中由调度器设置
bm.hash_blocks(s1)                         # 模拟 postprocess 的哈希登记
s2 = Sequence(list(range(6)) + [99, 100])  # 前 6 个 token 相同 -> 首块完全一致
assert bm.can_allocate(s2) == 1            # 命中 1 个完整前缀块
bm.allocate(s2, 1)
print('shared prefix ok, cached tokens =', s2.num_cached_tokens)"   # 4
```

(前两处手动设置对应真实引擎里 `LLMEngine.__init__` 与 `Scheduler` 的职责;漏掉它们 `can_allocate` 会返回 0——本指南写作时实测踩过。)
