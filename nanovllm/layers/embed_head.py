import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            # 将经过 tokenizer 分词的输入张量[seq_len, 1]，计算其各个位置的 token 的 id 是否属于当前 rank 负责的词表范围；如果属于，将其位置的掩码置为 1；否则置为 0
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 使用掩码屏蔽当前 rank 负责词表范围之外的 token（使其 token id 为 0，实际只是让这些范围外的 token 都嵌入到 id 为 0 的 embedding 上）
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # mask.shape = [seq_len,]
            # mask.unsqueeze(1).shape = [seq_len, 1]
            # y.shape = [embedding_dim, seq_len]
            # 使用掩码屏蔽当前 rank 负责词表范围之外的 token 的 embedding（使其 dim=1 维度的值为 0）
            y = mask.unsqueeze(1) * y
            # 各个 rank 的 embedding 矩阵都为[seq_len, embedding_dim]，只是各个 rank 只完成了各自负责的词表范围的 embedding 计算，所以需要将各个 rank 的 embedding 矩阵叠加（使用 all_reduce），得到完整的 embedding 矩阵
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # [Q]什么时候填充的 cu_seqlens?从代码上看，感觉就是获取[seq_len,hidden_size]的矩阵的最后一维
            last_indices = context.cu_seqlens_q[1:] - 1
            # [Q]contiguous 是 torch 的什么语法？
            x = x[last_indices].contiguous()
        # x.shape = [hidden_size,]
        # self.weight.shape = [num_embeddings, hidden_size]
        # self.weight.shape = [num_embeddings_per_partition, hidden_size]
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
