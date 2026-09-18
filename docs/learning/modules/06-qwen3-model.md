# 06 · Qwen3 模型与权重加载

**文件:** [`nanovllm/models/qwen3.py`](../../../nanovllm/models/qwen3.py) · [`nanovllm/utils/loader.py`](../../../nanovllm/utils/loader.py)

一个手写(而非 `from_pretrained`)的 Qwen3 解码器,唯一支持的模型架构。它把模块 [03](03-compiled-kernels.md) 的内核与模块 [05](05-tensor-parallel-layers.md) 的并行层组装成完整前向;`loader.py` 则演示了 HF safetensors 权重如何被"改名 + 分片"塞进这些自定义模块。

## 前置知识

- **decoder-only Transformer 解剖** — embedding → N×(归一化→注意力→归一化→MLP,残差贯穿)→ 终归一化 → LM head;这是读懂 `Qwen3Model.forward` 的全部框架。
- **Qwen3 架构特性** — GQA(28 q 头 / 8 kv 头)、**QK-Norm**(对 q/k 各做一次 RMSNorm)、RoPE、tied embeddings(0.6B 开启)。
- **safetensors API** — `safe_open` 按 key 流式读取权重文件,不再整包载入内存。
- **`getattr(config, '...', 默认值)` 防御式读法** — Qwen3Config 未必有某些字段(如 `attention_bias`),代码大量使用带默认值的读取。

## 运行时位置

`ModelRunner.__init__` 构造 `Qwen3ForCausalLM(hf_config)` 并随后 `load_model`(`nanovllm/engine/model_runner.py:31-32`)。它只暴露两个入口:`forward`(返回隐状态)与 `compute_logits`(接 LM head)——这个拆分正是 CUDA graph 只重放主干、logits 单独计算的前提(见模块 [09](09-model-runner.md))。

## 从哪里开始读

自底向上:`Qwen3Attention` → `Qwen3MLP` → `Qwen3DecoderLayer` → `Qwen3Model` → `Qwen3ForCausalLM`,最后读 `loader.py`(60 行)。

## 核心对象与 API

| 对象 | 位置 | 作用 |
|---|---|---|
| `Qwen3ForCausalLM` | `nanovllm/models/qwen3.py:186` | 顶层;持有 `packed_modules_mapping` |
| `Qwen3Model` | `nanovllm/models/qwen3.py:162` | embed → layers → norm 主干 |
| `Qwen3DecoderLayer` | `nanovllm/models/qwen3.py:120` | 双 RMSNorm + attention + MLP,残差融合 |
| `Qwen3Attention` | `nanovllm/models/qwen3.py:14` | 融合 QKV 投影 → QK-Norm → RoPE → flash-attn → o_proj |
| `Qwen3MLP` | `nanovllm/models/qwen3.py:91` | gate_up(融合列并行)→ SiluAndMul → down(行并行) |
| `packed_modules_mapping` | `nanovllm/models/qwen3.py:187-193` | HF 权重名 → (融合参数名, 分片 id) 映射表 |
| `load_model` | `nanovllm/utils/loader.py:12` | 遍历 `*.safetensors`,按映射改名并调用 `weight_loader` |

## 关键流程

### 前向:残差流的完整走线

`Qwen3DecoderLayer.forward`(`nanovllm/models/qwen3.py:146-159`)是全文件的心脏:

1. 第一层 `residual is None`,特殊分支:归一化输入、原输入直接当残差(`:152-153`);
2. 之后每层 `input_layernorm(hidden, residual)` 一步完成"加残差 + 归一化"(模块 [03](03-compiled-kernels.md) 的融合 kernel);
3. attention 与 MLP 之间不显式加残差——`post_attention_layernorm(hidden, residual)` 返回的新二元组已把加法融合进去;
4. `Qwen3Model.forward`(`:173-183`)循环所有层,最后 `self.norm(hidden_states, residual)` 收尾。

`Qwen3Attention.forward`(`:72-88`)的 shape 变化值得手推一遍:`qkv_proj` 出 `(n_tok, (q+2kv)*D)` → `split` 成三段 → `view(-1, heads, D)` → QK-Norm/RoPE → `Attention`(模块 [04](04-attention-and-context.md))→ `flatten` → `o_proj`。

### 权重加载:改名 + 分片两步走

HF checkpoint 的名字是 `q_proj`/`k_proj`/`v_proj`/`gate_proj`/`up_proj`,而模型里是融合后的 `qkv_proj`/`gate_up_proj`。`load_model`(`nanovllm/utils/loader.py:12-28`)的处理:

1. 对每个权重名,查 `packed_modules_mapping`:命中则改名为融合参数、带上 `shard_id`(`"q"/"k"/"v"` 或 `0/1`)调用参数的 `weight_loader`(`:17-24`);
2. 未命中(如 `o_proj`、`embed_tokens`、各 norm)走 `for-else` 分支,用参数自带的 `weight_loader` 或 `default_weight_loader` 直接拷贝(`:25-28`);
3. **每个参数自己决定怎么切**——这正是模块 [05](05-tensor-parallel-layers.md) 的 `weight_loader` 协议;加载器对 TP 完全无感。

### tied embeddings 的实现

`Qwen3ForCausalLM.__init__`(`:200-203`):`lm_head.weight.data = model.embed_tokens.weight.data`——直接**别名同一块存储**。注意两者都仍是词表并行的分片,所以权重加载时两处会被写入同一数据(各写各的 shard,结果一致)。

## 常见疑问 / 坑

- `forward` 与 `compute_logits` 分离不是风格问题:CUDA graph 重放只覆盖主干(见 `nanovllm/engine/model_runner.py:198` 的 eager 路径对比 `:212` 的 graph 路径),logits 投影在 graph 外做,因为它依赖"每序列最后 token"的动态索引。
- `q_norm`/`k_norm` 只在 `qkv_bias` 为假时创建(`:68-70`)——Qwen3 默认无 bias,所以正常路径都有 QK-Norm;这与论文/Qwen3 代码一致,但与很多其他 Llama 系模型不同。
- **只支持 Qwen3 一种架构**:换模型家族需要新写一个 `models/xxx.py` 并在 `model_runner.py:31` 替换引用——没有模型注册机制。

## 验证

用本机权重做一次纯 CPU/单卡加载检查(不跑推理):

```bash
MODEL=/home/tiger/.cache/modelscope/models/Qwen--Qwen3-0.6B/snapshots/master
.venv/bin/python -c "
from transformers import AutoConfig
from nanovllm.models.qwen3 import Qwen3ForCausalLM
import torch.distributed as dist
dist.init_process_group('gloo', 'tcp://localhost:39517', world_size=1, rank=0)
cfg = AutoConfig.from_pretrained('$MODEL')
m = Qwen3ForCausalLM(cfg)
print(sum(p.numel() for p in m.parameters())/1e6, 'M params')"
# 输出 ~751.6 M params(Qwen3-0.6B 含 embedding 的总量);端口被占就换一个空闲端口
```
(完整推理验证见 [README](../README.md#常用工作流)。)
