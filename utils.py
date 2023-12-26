import torch
from torch import distributed as dist


def reduce_tensor(tensor) -> torch.Tensor:
    tensor_ = tensor.clone()
    dist.all_reduce(tensor_, op=dist.ReduceOp.SUM)
    tensor_ /= dist.get_world_size()
    return tensor_
