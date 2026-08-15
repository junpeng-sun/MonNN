import torch
import torch.nn as nn
from typing import List
import random
from itertools import combinations

def get_pairs(data: torch.Tensor, max_n_pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate pairs of data points for interpolation.

    Args:
        data (torch.Tensor): Input data tensor.
        max_n_pairs (int): Maximum number of pairs to generate.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Two tensors containing the left and right pairs.
    """
    if len(data) < 2:
        return data, data

    all_pairs = list(combinations(range(len(data)), 2))
    if len(all_pairs) > max_n_pairs:
        all_pairs = random.sample(all_pairs, max_n_pairs)
    all_pairs = torch.LongTensor(all_pairs).to(data.device)

    pairs_left = torch.index_select(data, 0, all_pairs[:, 0])
    pairs_right = torch.index_select(data, 0, all_pairs[:, 1])

    return pairs_left, pairs_right


def interpolate_pairs(pairs: tuple[torch.Tensor, torch.Tensor], interpolation_range: float = 0.5) -> torch.Tensor:
    """
    Perform linear interpolation between pairs of data points.

    Args:
        pairs (tuple[torch.Tensor, torch.Tensor]): Tuple of left and right pairs.
        interpolation_range (float): Range around 0.5 for interpolation factors.

    Returns:
        torch.Tensor: Interpolated data points.
    """
    pairs_left, pairs_right = pairs
    lower_bound = 0.5 - interpolation_range
    upper_bound = 0.5 + interpolation_range
    interpolation_factors = torch.rand(len(pairs_left), 1, device=pairs_left.device) * (upper_bound - lower_bound) + lower_bound
    return interpolation_factors * pairs_left + (1 - interpolation_factors) * pairs_right


def mixupPWL_mono_reg(
    model: nn.Module,
    x: torch.Tensor,
    monotonic_indices: List[int],
    interpolation_range: float = 0.5,
    use_random: bool = True,
    max_n_pairs: int = 1024,
) -> torch.Tensor:
    """
    Compute monotonicity regularization loss using pairwise linear (PWL) interpolation.

    Args:
        model (nn.Module): The neural network model.
        x (torch.Tensor): Input data tensor.
        monotonic_indices (List[int]): Indices of features that should be monotonic.
        interpolation_range (float): Range around 0.5 for interpolation factors.
        use_random (bool): Whether to use random data points in addition to input data.

    Returns:
        torch.Tensor: Monotonicity regularization loss.
    """
    if not monotonic_indices:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if max_n_pairs < 1:
        raise ValueError("max_n_pairs must be positive")

    if use_random:
        # Inputs in the experiments are scaled from the outer-training fold to
        # [0, 1], which is the domain used by the original sampling protocol.
        random_data = torch.rand_like(x)
        combined_data = torch.cat([x, random_data], dim=0)
    else:
        combined_data = x

    # Generate pairs and interpolate
    pairs = get_pairs(combined_data, max_n_pairs=max_n_pairs)
    reg_points = interpolate_pairs(pairs, interpolation_range)

    # Create mask for monotonic features
    monotonic_mask = torch.zeros(
        reg_points.shape[1], dtype=torch.bool, device=reg_points.device
    )
    monotonic_mask[monotonic_indices] = True
    reg_points_monotonic = reg_points[:, monotonic_mask]
    reg_points_monotonic.requires_grad_(True)

    # Prepare input for the model
    reg_points_grad = reg_points.clone()
    reg_points_grad[:, monotonic_mask] = reg_points_monotonic

    # Forward pass
    y_pred_m = model(reg_points_grad)

    # Compute gradients and monotonicity loss
    grads = torch.autograd.grad(y_pred_m.sum(), reg_points_monotonic, create_graph=True)[0]
    monotonicity_term = torch.relu(-grads).square()
    return monotonicity_term.sum(dim=1).mean()
