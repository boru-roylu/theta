import torch
import torch.distributed as dist


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return -1
    if not dist.is_initialized():
        return -1
    return dist.get_rank()


def get_device():
    if not dist.is_available() or not dist.is_initialized():
        return torch.device('cuda', 0)
    else:
        return torch.device('cuda', get_rank())


def is_local_master():
    return get_rank() in [-1, 0]


def is_distributed():
    return get_rank() != -1


def wait_master():
    if is_local_master():
        dist.barrier()


def wait_others():
    if not is_local_master():
        dist.barrier()


def wait_all():
    dist.barrier()