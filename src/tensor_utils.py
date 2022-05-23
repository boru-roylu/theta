import numpy as np
import torch
import torch.distributions.binomial as binomial


def masked_mean(tensor, mask, dim):
    masked = torch.mul(tensor, mask)  # Apply the mask using an element-wise multiply
    return masked.sum(dim=dim) / mask.sum(dim=dim)  # Find the average!


def cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    torch.cdist is implemented for 'Half'.
    """
    if x.dtype is torch.float16 and x.is_cuda:
        # einops is slower, so we use unsqueeze instead.
        # import einops
        #x = einops.rearrange(x, "b l r -> b l () r")
        #y = einops.rearrange(y, "b l r -> b () l r")
        x = x.unsqueeze(2)
        y = y.unsqueeze(1)
        return (x - y).norm(dim=-1, p=2)
    return torch.cdist(x, y, p=2)


def batched_index_select(tensor, dim, index):
    assert (index >= 0).all()
    views = [tensor.shape[0]] + \
    	[1 if i != dim else -1 for i in range(1, tensor.ndim)]
    expanse = list(tensor.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(tensor, dim, index)


def np_batched_index_select(arr, dim, index):
    tensor = torch.from_numpy(arr)
    index = torch.from_numpy(index)
    target = batched_index_select(tensor, dim, index)
    return target.numpy()


def batched_p_norm(x1, x2, temperature):
    channels = x1.size(-1)
    d = -torch.square(cdist(x1, x2))
    d /= channels
    d /= temperature
    return d


def split_tensor_by_idxs(tensor, idxs):
    split_tensor = torch.split(tensor, idxs.tolist())
    return split_tensor


def pad_tensor(t, max_seq_len, pad_value, dim=-1, pad_left=False):
    seq_len = t.size(dim)
    pad_len = max_seq_len - seq_len
    assert pad_len >= 0

    if dim == -1:
        if pad_left:
            padding = (pad_len, 0)
        else:
            padding = (0, pad_len)
    else:
        padding = []
        for _ in range(t.ndim - dim - 1):
            padding.extend([0, 0])
        if pad_left:
            padding.extend([pad_len, 0])
        else:
            padding.extend([0, pad_len])

    t = torch.nn.functional.pad(t, padding, value=pad_value)
    return t


def binomial_sample(t1, t2, prob, index=None):
    """Binomial sample on two tensors.

    Args:
        t1 (torch.Tensor): A 1-dim torch Tensor.
        t2 (torch.Tensor): A 1-dim torch Tensor.
        prob (float): Probability of binomial distribution. Larger value, higher
            chance to sample from t2.
        index (torch.Tensor): Uses the predefined indices to sample.

    Returns:
        t: torch.Tensor: Sampled tensor.
        x: torch.Tensor: Sampled index. 
    """
    device = t1.device
    dtype = t1.dtype
    t = torch.stack((t1, t2))
    t = t.T
    if index is None:
        m = binomial.Binomial(torch.ones(t1.size(0)), probs=prob)
        x = m.sample().to(device=device, dtype=dtype)
    else:
        x = index
    t = batched_index_select(t, 1, x).view(-1)
    t = t.to(device)
    return t, x


def np_softmax(a, axis=None):
    """
    Computes exp(a)/sumexp(a); relies on scipy logsumexp implementation.
    :param a: ndarray/tensor
    :param axis: axis to sum over; default (None) sums over everything
    """
    from scipy.special import logsumexp
    lse = logsumexp(a, axis=axis)  # this reduces along axis
    if axis is not None:
        lse = np.expand_dims(lse, axis)  # restore that axis for subtraction
    return np.exp(a - lse)