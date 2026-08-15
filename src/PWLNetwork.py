from typing import List

import torch
import torch.nn as nn


def pwl_mono_reg(
    model: nn.Module,
    x: torch.Tensor,
    monotonic_indices: List[int],
) -> torch.Tensor:
    """
    Compute Point wise monotonicity regularization loss for a neural network.

    This function calculates a regularization term that encourages monotonicity
    in the specified input dimensions of the model's output.

    Args:
        model (nn.Module): The neural network model to regularize.
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        monotonic_indices (List[int]): Indices of input dimensions that should be monotonic.
    Returns:
        torch.Tensor: The computed monotonicity regularization loss.
    """
    if not monotonic_indices:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    x_grad = x.detach().clone().requires_grad_(True)
    y_pred_m = model(x_grad)
    grads = torch.autograd.grad(y_pred_m.sum(), x_grad, create_graph=True)[0]

    # Gupta et al. (2019), Eq. (1): penalize the negative divergence over the
    # designated feature set.  The mean is the mini-batch Monte-Carlo form of
    # that empirical objective and keeps its scale independent of batch size.
    divergence = grads[:, monotonic_indices].sum(dim=1)
    return torch.relu(-divergence).mean()
