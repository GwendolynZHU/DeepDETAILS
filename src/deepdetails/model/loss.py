import torch
from torch import nn


class RMSLELoss(nn.Module):
    """
    Root Mean Square Log Error

    """
    def __init__(self, squared=True):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")
        self.squared = squared

    def forward(self, pred: torch.Tensor, actual: torch.Tensor):
        # shape of self.mse(): batch, n_targets, seq_len
        # shape of self.mse().sum(axis=-1): batch, n_targets
        mse = self.mse(torch.log(torch.clamp_min(pred, -0.999) + 1), torch.log(actual + 1)).sum(axis=-1).mean()
        return torch.sqrt(mse) if self.squared else mse


def off_diagonal(x: torch.Tensor):
    """Return a flattened view of the off-diagonal elements of a square matrix

    Parameters
    ----------
    x : torch.Tensor
        A square matrix (correlation matrix)
        Shape: n, m, where n==m

    Returns
    -------
    torch.Tensor
        n*n-n
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
