# 04 · Attention 层与运行时 Context

**文件:** [`nanovllm/layers/attention.py`](../../../nanovllm/layers/attention.py) · [`nanovllm/utils/context.py`](../../../nanovllm/utils/context.py)

这一对文件是全仓库**数据结构**(分页 KV cache)与**算子**(flash-attention)的交汇点,也是理解"prefill 与 decode 为什么走两条完全不同代码路径"的最佳入口。`utils/context.py` 是一个 27 行的全局单例,却是贯穿模型与执行器的隐形传参通道。

## 前置知识

- **flash-attention 两个 API** — `flash_attn_varlen_func`(变长打包序列的 prefill)与 `flash_attn_with_kvcache`(从 KV 缓存做单步解码);不熟悉可先看它们的签名,参数与下文一一对应。
- **分页 KV cache(vLLM PagedAttention 的核心思想)** — KV 按 block 存储,序列通过 `block_table`(block id 列表)间接寻址;物理连续性不再必要。
- **GQA(分组查询注意力)** — k/v 头数少于 q 头数(Qwen3-0.6B:28 q 头 / 8 kv 头);flash-attn 原生支持广播。
- **Triton kernel 基础** — `store_kvcache_kernel` 是全仓库唯一的 Triton kernel,一个 program 搬一行,读懂它不需要深入 Triton。

## 运行时位置

模型每一层都有一个 `Attention` 实例(挂 `k_cache`/`v_cache` 视图,由模块 [09](09-model-runner.md) 在启动时注入)。它每次 forward 都从 `get_context()` 读取本 step 的元数据——context 由 `ModelRunner` 在**每次模型调用前**写入(`nanovllm/engine/model_runner.py:169,187`),调用后重置(`nanovllm/engine/model_runner.py:219`)。

## 从哪里开始读

先读 27 行的 `utils/context.py` 全文,再读 `attention.py` 的 `Attention.forward`(`nanovllm/layers/attention.py:59-75`),最后回头看 Triton kernel。

## 核心对象与 API

| 对象 | 位置 | 作用 |
|---|---|---|
| `Context` dataclass | `nanovllm/utils/context.py:5-14` | 本 step 的全部元数据:prefill 标志、`cu_seqlens_q/k`、`slot_mapping`、`context_lens`、`block_tables` |
| `set_context` / `get_context` / `reset_context` | `nanovllm/utils/context.py:21-27` | 模块级全局读写的三件套 |
| `Attention` | `nanovllm/layers/attention.py:43` | 每层一个;持有该层的 `k_cache`/`v_cache` 视图 |
| `store_kvcache_kernel` | `nanovllm/layers/attention.py:10-30` | Triton:把本 step 的 k/v 行散写进分页缓存 |
| `flash_attn_varlen_func` | `nanovllm/layers/attention.py:67-70` | prefill:打包变长序列 + 可选 block_table |
| `flash_attn_with_kvcache` | `nanovllm/layers/attention.py:72-74` | decode:单 token 查询整段缓存 |

## 关键流程

### 写缓存:Triton kernel

`store_kvcache`(`nanovllm/layers/attention.py:33-40`)把 `(N, num_heads, head_dim)` 的 k/v 展平成每行 `D = num_heads*head_dim`,按 `slot_mapping[i]` 写入缓存的第 `slot` 行。两个细节值得咀嚼:

- **`slot == -1` 直接跳过**(`nanovllm/layers/attention.py:23`)——这是给 CUDA graph 用的:CUDA graph 重放时 batch 被填充到桶大小,多余行的 slot 填 `-1`(见 `nanovllm/engine/model_runner.py:206`),kernel 静默丢弃,避免污染缓存。
- 写缓存发生在 `if` 判断之外——**prefill 和 decode 都要写**,只是读的路径不同。

### 读:prefill 与 decode 分道

**Prefill**(`nanovllm/layers/attention.py:64-70`):
- 常规 prefill:`flash_attn_varlen_func` 直接吃刚算出来的 `q, k, v`,各序列拼接成一维打包张量,`cu_seqlens_q/k` 描述边界,casual mask 由 kernel 内部处理——**batch 内变长序列零 padding**。
- **前缀缓存命中时**:`block_tables is not None` 分支把 `k, v` 换成整块缓存(`nanovllm/layers/attention.py:65-66`),flash-attn 用 `block_table` 间接寻址,新 token 只需attend缓存里已有前缀 + 自己。

**Decode**(`nanovllm/layers/attention.py:71-74`):
- 每序列只有 1 个新 token,`q.unsqueeze(1)` 变成 `(bs, 1, H, D)`,直接对 `k_cache/v_cache` 做 `flash_attn_with_kvcache`:`cache_seqlens=context_lens` 告诉每条序列看多长,`block_table` 提供分页寻址。

### 为什么需要全局 context?

`Attention` 和 `ParallelLMHead`(模块 [05](05-tensor-parallel-layers.md))深埋在模型第 N 层里,却需要"本 step 是 prefill 还是 decode、序列边界在哪"这类执行器信息。若层层传参会污染所有 `nn.Module` 签名。于是 `ModelRunner` 每步 `set_context(...)`,模型深处 `get_context()` 自取——本质是**用模块级全局变量换干净的函数签名**,代价是必须保证设置/重置成对出现(`run()` 末尾 `reset_context()`,`nanovllm/engine/model_runner.py:219`)。

## 常见疑问 / 坑

- `k_cache` 初始为 `torch.tensor([])`(`nanovllm/layers/attention.py:57`),真正的缓存视图由 `allocate_kv_cache` 注入;`forward` 里 `if k_cache.numel()` 的判断就是防"未注入就前向"。
- 变长打包意味着 **batch 维不存在**:prefill 的 q 是 `(总token数, H, D)`,序列边界全靠 `cu_seqlens`。调试 shape 时别按 `(B, S, H, D)` 想。

## 验证

跑通 [README 的 example](../README.md#常用工作流) 即覆盖此模块(prefill+decode+缓存写入全链路)。
