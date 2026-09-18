# 05 · 张量并行投影层

**文件:** [`nanovllm/layers/linear.py`](../../../nanovllm/layers/linear.py) · [`nanovllm/layers/embed_head.py`](../../../nanovllm/layers/embed_head.py)

Megatron 风格张量并行(TP)的最小实现:四种 Linear 变体 + 并行 Embedding/LM head。除了矩阵切法本身,这里还藏着一个贯穿全仓库的**权重加载协议**(`param.weight_loader`),模块 [06](06-qwen3-model.md) 的加载器完全建立在它之上。

## 前置知识

- **Megatron 式张量并行** — Column-Parallel 沿**输出维**切(各 rank 得到不同输出列,无需通信);Row-Parallel 沿**输入维**切(各 rank 得到部分和,最后 `all_reduce` 求和)。两层背靠背组合即完成一个 attention/MLP 块的切分。
- **`torch.distributed` 集合通信** — `dist.get_rank()/get_world_size()` 在这里被当作"当前进程是几号卡"来用;`all_reduce`、`gather` 是仅有的两个通信原语。
- **GQA 的权重布局** — Q/K/V 拼在同一权重里,kv 段比 q 段短;`QKVParallelLinear` 的 offset 计算依赖这一布局。
- **`get_context()`**(模块 [04](04-attention-and-context.md))— `ParallelLMHead` 用它区分 prefill/decode。

## 运行时位置

模型构建前必须先 `dist.init_process_group`——`LinearBase.__init__` 直接调 `dist.get_rank()`(`nanovllm/layers/linear.py:23-24`),这正是 `ModelRunner` 先初始化进程组再建模型的原因(`nanovllm/engine/model_runner.py:26-31`)。每层 qkv/o/gate_up/down 投影都用这里的类。

## 从哪里开始读

`linear.py` 里的 `LinearBase`(`nanovllm/layers/linear.py:12-31`)→ `ColumnParallelLinear` → `RowParallelLinear`,然后回头读两个变体;`embed_head.py` 全文只有 66 行。

## 核心对象与 API

| 对象 | 位置 | 作用 |
|---|---|---|
| `LinearBase` | `nanovllm/layers/linear.py:12-31` | 公共底座:创建空权重并挂 `weight_loader` 属性 |
| `ColumnParallelLinear` | `nanovllm/layers/linear.py:54-73` | 输出维按 rank 切;`tp_dim=0` |
| `RowParallelLinear` | `nanovllm/layers/linear.py:131-156` | 输入维按 rank 切;forward 末尾 `all_reduce`(`:155`) |
| `MergedColumnParallelLinear` | `nanovllm/layers/linear.py:76-93` | gate/up 两块逻辑权重熔进一个参数,按 `shard_id` 加载 |
| `QKVParallelLinear` | `nanovllm/layers/linear.py:96-128` | q/k/v 三段融合,GQA 下 kv 段更短 |
| `VocabParallelEmbedding` | `nanovllm/layers/embed_head.py:9-42` | 词表维切分;越界 id 置零 + `all_reduce` 聚合 |
| `ParallelLMHead` | `nanovllm/layers/embed_head.py:45-66` | LM head:prefill 只取每序列最后 token;词表分片 gather 到 rank 0 |

## 关键流程

### weight_loader 协议(全仓库的加载约定)

`LinearBase` 给每个参数挂上 `param.weight_loader = self.weight_loader`(`nanovllm/layers/linear.py:26`)。加载器(模块 [06](06-qwen3-model.md))遍历 checkpoint 时**不关心切分方式**,只按参数名找到参数、调用它的 `weight_loader(param, tensor)`。各层各自决定如何从完整权重里抠出自己那份:

- `ColumnParallelLinear.weight_loader`:按 `tp_rank * shard_size` 偏移 `narrow` 出本 rank 的列(`nanovllm/layers/linear.py:65-70`);
- `QKVParallelLinear.weight_loader`:接受 `loaded_shard_id in {"q","k","v"}`,按各段头数算 offset 写进融合参数的对应区段(`nanovllm/layers/linear.py:114-128`)。

### Row-Parallel 的通信时机

`RowParallelLinear.forward`(`nanovllm/layers/linear.py:152-156`):各 rank 算出**部分和**,`F.linear` 后 `dist.all_reduce(y)` 求和;bias 只加一次(rank 0)以避免重复加。这解释了为什么 `o_proj`、`down_proj` 必须是 Row-Parallel——它们前面正好是按列切分的输入。

### LM head 的两个特殊动作

`ParallelLMHead.forward`(`nanovllm/layers/embed_head.py:56-66`):

1. **prefill 时丢弃非末位 token**:logits 只需要每个序列的最后一个位置,`x = x[cu_seqlens_q[1:] - 1]` 用累计序列长度直接索引(`:58-60`)——省掉整个 batch 的 vocab 投影。
2. **词表并行聚合**:各 rank 持有词表的一段,logits 形状 `(n_tok, vocab/tp)`,经 `dist.gather` 到 rank 0 后沿最后一维 `cat` 成完整 logits(`:62-65`)。

`VocabParallelEmbedding.forward` 则是掩码套路:不属于本 rank 词表段的 id 置 0,查表后把非本段行清零,`all_reduce` 求和得到完整 embedding(`:34-42`)。

## 常见疑问 / 坑

- `ColumnParallelLinear` 的 `weight_loader` 与 `QKVParallelLinear` 的**签名不同**(后者多一个 `shard_id` 必填参数)——加载器靠 `packed_modules_mapping` 知道该传什么(见模块 [06](06-qwen3-model.md))。
- `tp_dim`(0=列切,1=行切)只在 `weight_loader` 里用于 `narrow`,forward 并不读它。
- 所有 TP 通信都是**同步**的:rank 间靠"执行相同的 call 序列 + 集合通信"保持一致,没有显式的控制流同步(控制面同步见模块 [09](09-model-runner.md) 的 shm 机制)。

## 验证

TP=1 时这些层退化为普通 Linear/Embedding,单卡即可跑通;多卡路径见 [README 工作流](../README.md#常用工作流)的 `tensor_parallel_size=2`(需 ≥2 GPU,本机 1 卡,未验证)。
