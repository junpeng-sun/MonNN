import csv
import json
import itertools
from pathlib import Path
from typing import List, Literal, Union, Dict, Optional

import torch
from torch import nn


# Monotonicity Check
def monotonicity_check(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_x: torch.Tensor,
    monotonic_indices: List[int],
    device: torch.device
) -> float:
    """
    Compute the fraction of points that violate coordinatewise monotonicity.

    A point is counted as a violation if df/dx_i < -1e-8 for at least one
    constrained feature. ``optimizer`` is retained for API compatibility but
    is deliberately not used: this audit must not alter parameter gradients.
    """

    if not monotonic_indices:
        return 0.0

    model.eval()

    data_x = data_x.detach().clone().to(device).requires_grad_(True)
    n_points = data_x.shape[0]
    if n_points == 0:
        return 0.0

    outputs = model(data_x)
    grads = torch.autograd.grad(
        outputs.sum(), data_x, create_graph=False, retain_graph=False
    )[0]
    constrained_grads = grads[:, monotonic_indices]
    violations = (constrained_grads < -1e-8).any(dim=1)
    return float(violations.float().mean().item())


# Reordered Monotonic Indices
def get_reordered_monotonic_indices(dataset_name: str) -> List[int]:
    from dataPreprocessing.loaders import get_monotonic_feature_count

    num = get_monotonic_feature_count(dataset_name)
    return list(range(num))


# Monotonicity Indicator
def create_monotonicity_indicator(
    monotonic_indices: List[int],
    input_size: int
) -> List[int]:

    indicator = [0] * input_size

    for idx in monotonic_indices:
        if 0 <= idx < input_size:
            indicator[idx] = 1

    return indicator


# Weight Initialization
def init_weights(
    module_or_tensor: Union[nn.Module, torch.Tensor],
    method: Literal[
        'xavier_uniform',
        'xavier_normal',
        'kaiming_uniform',
        'kaiming_normal',
        'he_uniform',
        'he_normal',
        'truncated_normal',
        'uniform',
        'zeros'
    ],
    **kwargs
) -> None:

    def init_tensor(tensor):

        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(tensor)

        elif method == 'xavier_normal':
            nn.init.xavier_normal_(tensor)

        elif method in ['kaiming_uniform', 'he_uniform']:
            nn.init.kaiming_uniform_(tensor)

        elif method in ['kaiming_normal', 'he_normal']:
            nn.init.kaiming_normal_(tensor)

        elif method == 'truncated_normal':
            mean = kwargs.get('mean', 0.)
            std = kwargs.get('std', 1.)

            with torch.no_grad():
                tensor.normal_(mean, std)

                while True:
                    cond = (
                        (tensor < mean - 2 * std) |
                        (tensor > mean + 2 * std)
                    )
                    if not torch.sum(cond):
                        break
                    tensor[cond] = tensor[cond].normal_(mean, std)

        elif method == 'uniform':
            a = kwargs.get('a', 0.)
            b = kwargs.get('b', 1.)
            nn.init.uniform_(tensor, a=a, b=b)

        elif method == 'zeros':
            nn.init.zeros_(tensor)

        else:
            raise ValueError(f"Unsupported initialization method: {method}")

    if isinstance(module_or_tensor, nn.Module):
        for param in module_or_tensor.parameters():
            init_tensor(param)

    elif isinstance(module_or_tensor, torch.Tensor):
        init_tensor(module_or_tensor)

    else:
        raise TypeError("Input must be nn.Module or torch.Tensor")


# Positive Weight Transform
def transform_weights(
    module_or_tensor: Union[nn.Module, torch.Tensor],
    method: Literal['exp', 'explin', 'sqr']
):

    def transform_tensor(tensor):

        if method == 'exp':
            return torch.exp(tensor)

        elif method == 'explin':
            return torch.where(
                tensor > 1.,
                tensor,
                torch.exp(tensor - 1.)
            )

        elif method == 'sqr':
            return torch.square(tensor)

        else:
            raise ValueError(f"Unsupported transform method: {method}")

    if isinstance(module_or_tensor, nn.Module):
        return nn.ParameterList([
            nn.Parameter(transform_tensor(param))
            for param in module_or_tensor.parameters()
        ])

    elif isinstance(module_or_tensor, torch.Tensor):
        return transform_tensor(module_or_tensor)

    else:
        raise TypeError("Input must be nn.Module or torch.Tensor")


# CSV
def write_results_to_csv(
    filename: str | Path,
    dataset_name: str,
    task_type: str,
    metric_name: str,
    metric_mean: float,
    metric_std: float,
    secondary_metric_name: str,
    secondary_metric_mean: float,
    secondary_metric_std: float,
    n_params: int,
    best_config: Dict,
    mono_metrics: Optional[Dict]
):

    best_config_str = json.dumps(best_config)


    m_mean = f"{metric_mean:.4f}" if isinstance(metric_mean, (int, float)) else metric_mean
    m_std = f"{metric_std:.4f}" if isinstance(metric_std, (int, float)) else metric_std
    secondary_mean = (
        f"{secondary_metric_mean:.4f}"
        if isinstance(secondary_metric_mean, (int, float))
        else secondary_metric_mean
    )
    secondary_std = (
        f"{secondary_metric_std:.4f}"
        if isinstance(secondary_metric_std, (int, float))
        else secondary_metric_std
    )


    row = [
        dataset_name,
        task_type,
        metric_name,
        m_mean,
        m_std,
        secondary_metric_name,
        secondary_mean,
        secondary_std,
        n_params,
        best_config_str
    ]


    if mono_metrics is not None:
        for key in ['random', 'train', 'test']:
            mean, std = mono_metrics.get(key, (0.0, 0.0))
            row.extend([f"{mean:.4f}", f"{std:.4f}"])

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


# Parameter Counter
def count_parameters(module: nn.Module) -> int:
    return sum(
        p.numel() for p in module.parameters()
        if p.requires_grad
    )


# Layer Combination Generator
def generate_layer_combinations(
    min_layers=1,
    max_layers=3,
    units=[8, 16, 32, 64]
):

    combinations = []

    for n_layers in range(min_layers, max_layers + 1):
        for combo in itertools.product(units, repeat=n_layers):
            combinations.append(list(combo))

    return [str(combo) for combo in combinations]
