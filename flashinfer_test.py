import torch
import flashinfer
from tqdm import tqdm
# 模型配置参数
num_layers = 32          # Transformer 层数
num_qo_heads = 64        # Query 头数
num_kv_heads = 16        # Key/Value 头数 (GQA)
head_dim = 128           # 每个头的维度
max_num_pages = 128      # KV Cache 最大页数
page_size = 16           # 每页的 token 数

# 分配 128MB 的工作空间缓冲区，用于 FlashInfer 内部计算
workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")

# 初始化 BatchPrefillWithPagedKVCacheWrapper
# "NHD" 表示输入布局为 (Total_Tokens, Num_Heads, Head_Dim)
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace_buffer, "NHD"
)

# Batch 配置
batch_size = 7           # Batch 大小
nnz_qo = 100             # Query 的总 token 数 (所有请求的 query 长度之和)

# Query 的索引指针 (indptr)，用于指示每个请求在 query 张量中的起始位置
# 格式: [0, len1, len1+len2, ..., sum(lens)]
qo_indptr = torch.tensor(
    [0, 33, 44, 55, 66, 77, 88, nnz_qo], dtype=torch.int32, device="cuda:0"
)

# Paged KV Cache 的页索引
paged_kv_indices = torch.arange(max_num_pages).int().to("cuda:0")

# Paged KV Cache 的页索引指针 (indptr)，用于指示每个请求占用的页数
paged_kv_indptr = torch.tensor(
    [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
)

# 每个请求最后一页的有效长度
# 约束: 1 <= paged_kv_last_page_len <= page_size
paged_kv_last_page_len = torch.tensor(
    [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
)

# 创建模拟的 Query 数据: (层数, 总Token数, Query头数, 头维度)
q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim).half().to("cuda:0")

# 创建模拟的 KV Cache 数据: (层数, 最大页数, 2(K/V), 页大小, KV头数, 头维度)
kv_cache_at_layer = torch.randn(
    num_layers, max_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device="cuda:0"
)

# 为 Batch Prefill Attention 创建辅助数据结构
# 这一步会计算并缓存必要的元数据，以便后续快速执行
prefill_wrapper.plan(
    qo_indptr=qo_indptr,
    paged_kv_indptr=paged_kv_indptr,
    paged_kv_indices=paged_kv_indices,
    paged_kv_last_page_len=paged_kv_last_page_len,
    num_qo_heads=num_qo_heads,
    num_kv_heads=num_kv_heads,
    head_dim_qk=head_dim,
    page_size=page_size,
    causal=True,  # 启用因果掩码 (Causal Masking)
)

outputs = []
# 遍历每一层进行计算
for i in tqdm(range(num_layers)):
    q = q_at_layer[i]
    kv_cache = kv_cache_at_layer[i]
    
    # 执行 Batch Prefill Attention 计算
    # 复用之前 plan() 创建的辅助数据结构
    o = prefill_wrapper.run(q, kv_cache)
    outputs.append(o)

# 打印第一层输出的形状，验证结果
print(outputs[0].shape)