# 03 · 编译内核层 — RMSNorm / SwiGLU / RoPE / Sampler

**文件:** [`nanovllm/layers/layernorm.py`](../../../nanovllm/layers/layernorm.py) · [`nanovllm/layers/activation.py`](../../../nanovllm/layers/activation.py) · [`nanovllm/layers/rotary_embedding.py`](../../../nanovllm/layers/rotary_embedding.py) · [`nanovllm/layers/sampler.py`](../../../nanovllm/layers/sampler.py)

四个被 `@torch.compile` 包住的小算子。它们本身只有几十行,却决定了两件事:残差流如何在线性层之间流动(模块 [06](06-qwen3-model.md) 的前向形态由它们决定),以及 logits 如何变成 token。单独读它们,能把 `torch.compile` 与解码采样这两个横切概念一次搞清。

## 前置知识

- **RMSNorm 公式** `x / sqrt(mean(x²) + ε) * w` — 比 LayerNorm 少一次均值中心化;Qwen3 全模型用它。
- **SwiGLU** — MLP 的门控激活:`silu(gate) * up`,对应 `SiluAndMul` 的"拆一半、乘一半"。
- **RoPE(旋转位置编码)** — 把 q/k 的每对分量按位置角度旋转;是 `positions` 张量唯一的消费者。
- **`torch.compile`** — 把 eager 小算子融合成单 kernel;本仓库所有内核都带此装饰器,CUDA graph 捕获时(模块 [09](09-model-runner.md))也会受益。
- **指数竞速采样** — `argmax(p_i / e_i)`(`e_i ~ Exp(1)`)等价于按类别分布 `p` 采样;是 `Sampler` 那一行魔法的理论依据。

## 运行时位置

- `RMSNorm`、`SiluAndMul`、`RotaryEmbedding` 在**每个前向、每层**都被调用(见 `nanovllm/models/qwen3.py:83,111,143-144`)。
- `Sampler` 每个 step 调用一次,且只在 TP rank 0 上执行(`nanovllm/engine/model_runner.py:218`)。
- `RotaryEmbedding` 经 `get_rope` 的 `lru_cache` 全局共享单例(`nanovllm/layers/rotary_embedding.py:51-59`)。

## 从哪里开始读

按 `activation.py`(11 行)→ `sampler.py`(12 行)→ `layernorm.py`(50 行)→ `rotary_embedding.py`(59 行)的顺序,由短到长。

## 核心对象与 API

| 对象 | 位置 | 作用 |
|---|---|---|
| `SiluAndMul.forward` | `nanovllm/layers/activation.py:8-11` | 最后一维对半拆分,silu 门控相乘(SwiGLU) |
| `Sampler.forward` | `nanovllm/layers/sampler.py:7-12` | 温度缩放 → softmax → 指数竞速采样 |
| `RMSNorm.rms_forward` | `nanovllm/layers/layernorm.py:17-26` | 标准路径;**先转 float32 计算,再转回原 dtype** |
| `RMSNorm.add_rms_forward` | `nanovllm/layers/layernorm.py:28-40` | 融合路径:同时完成 `x += residual`,返回 `(归一化值, 新残差)` 二元组 |
| `RotaryEmbedding` + `get_rope` | `nanovllm/layers/rotary_embedding.py:17-48,51-59` | 预计算 cos/sin 缓存;按 `positions` 查表旋转 q/k |
| `apply_rotary_emb` | `nanovllm/layers/rotary_embedding.py:6-14` | 前后半段拆分 → 旋转 → 拼回 |

## 关键流程

### 融合残差的 RMSNorm(理解模块 06 的钥匙)

Transformer 每层要写两次残差。传统写法是 `x = x + residual` 后再归一化,多一次全量读写。这里 `add_rms_forward(x, residual)` 一步返回 `(归一化后的 x, 相加后的 residual)`,decoder layer 直接串起两个 RMSNorm:

```
hidden, residual = input_layernorm(hidden, residual)   # 相加 + 归一化
hidden = attn(hidden)
hidden, residual = post_attention_layernorm(hidden, residual)
hidden = mlp(hidden)
```

`Qwen3DecoderLayer.forward`(`nanovllm/models/qwen3.py:146-159`)正是这个形态;`residual` 一路"穿透"层间,最后由 `Qwen3Model` 的 `self.norm(hidden_states, residual)` 收尾(`nanovllm/models/qwen3.py:182`)。

### Sampler 的采样数学

`nanovllm/layers/sampler.py:9-11`:

1. `logits / temperature` → softmax 得到概率 `p`;
2. `argmax(p / Exp(1))` —— 这就是**指数竞速**(Gumbel-trick 的指数版):给每个类别一个 `Exp(1)` 噪声,概率正比于 `p`。它与"按 p 采样"完全等价,却把采样压成一个 kernel,没有数据依赖的分支——这也是它能在 CUDA graph 里安全重放的原因。

## 常见疑问 / 坑

- **为什么 RMSNorm 内部强制 float32?** bf16/fp16 下 `mean(x²)` 容易溢出/精度不足;`rms_forward` 先 `x.float()` 再算,最后转回原 dtype(`nanovllm/layers/layernorm.py:21-25`)。改这段代码时保持这个习惯。
- **`RotaryEmbedding` 断言 `rotary_dim == head_size`**(`nanovllm/layers/rotary_embedding.py:28`)——部分旋转(partial RoPE)的模型(如部分 Qwen 系)无法直接用此实现。
- `get_rope` 的 `lru_cache(1)` 只按参数缓存一个实例;同进程内所有层共享同一 cos/sin 缓存,省显存也省编译时间。

## 验证

内核都经 `@torch.compile` 编译,建议在 GPU 上验证(与引擎实际运行路径一致):

```bash
.venv/bin/python -c "
import torch
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.layernorm import RMSNorm
torch.manual_seed(0)
print(Sampler().forward(torch.randn(2,16,device='cuda'), torch.tensor([0.6,0.6],device='cuda')).tolist())
print(RMSNorm(16)(torch.randn(2,16,device='cuda')).shape)"
```

只想验证算子逻辑(不触发编译)时可先 `torch._dynamo.config.disable = True`。**坑**:本环境(torch 2.x,2026-09)在纯 CPU 上编译 `Sampler` 会触发 inductor 的 `KeyError: 'buf1'` 崩溃(外层循环融合 bug);GPU 路径正常,端到端推理不受影响。
