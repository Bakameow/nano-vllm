import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator

# 定义LinearBase类，继承自nn.Module，用于实现线性层
class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# 每个 rank（进程）都拥有一份完整的、相同的参数副本，即参数复制（Replicated）
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # 调用父类LinearBase的__init__方法，初始化input_size、output_size、tp_dim、tp_rank、tp_size
        super().__init__(input_size, output_size)
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 对于 PyTorch 的 Module 对象，register_parameter 方法用于将参数注册到 Module 对象中
            # 注册后，参数可以被 Module 对象自动管理，例如在保存和加载模型时自动保存和加载参数
            # 如果只是使用 self.bias=None 来设置参数，则 bias 参数不会被注册，也就不会被自动管理
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# 完整的线性层的 shape 是[output_size, input_size]，在 dim=0 上切分，每个rank 输出output_size/tp_size 个分片
# 输入的shape 是[batch_size, input_size]，输出的shape 是[batch_size, output_size/tp_size]
class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 0)
        self.input_size_per_partition = input_size
        # 计算本rank需要输出的分片的维度，即output_size/tp_size
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # [Q]计算分片的大小
        shard_size = param_data.size(self.tp_dim)
        # 计算分片的起始索引
        start_idx = self.tp_rank * shard_size
        # 将 loaded_weight 按照 self.tp_dim=0 维度切分，并取 self.tp_rank 对应 rank 的参数
        # 使用narrow在指定dim=0维度上切片，start_idx是起始索引，shard_size是切片大小
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# [Q]不知道其在哪一层使用
# [A]在 FeedForward（MLP）层使用
class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

# 使用一个大矩阵，将 Wq、Wk、Wv 矩阵拼接起来，然后按照 dim=0 切分，每个 Rank 计算一部分，最后合并
class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        # Wq、Wk、Wv 矩阵就是多个同维度（相同 Shape）的线性层，这里直接将这三个矩阵拼接并划分到多个 rank 上
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        # 根据要加载的Wq、Wk、Wv 矩阵，计算出要加载的参数在param_data中的偏移量和大小
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 将 loaded_weight(可以是 W_q/q_proj、W_k/k_proj 或 W_v/v_proj 的参数)按照self.tp_dim 维度切分，并取self.tp_rank 对应 rank 的参数
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

# 完整的线性层的 shape 是[output_size, input_size]，在 dim=1 上切分，每个 rank 输出 output_size 维度的向量，但其输入仅接收 input_size/tp_size 特征
# 输入的shape 是[batch_size, input_size/tp_size]，输出的shape 是[batch_size, output_size]
class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, 1)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self.output_size_per_partition = output_size

        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
