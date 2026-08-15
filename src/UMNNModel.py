"""Conditional UMNN and generalized multivariate monotonic architecture.

This module is a PyTorch port of the public UMNN implementation.  It uses a
strictly positive ELU+1 integrand, a conditioning network for scale/offset,
Clenshaw--Curtis quadrature, and the nested construction from generalized
UMNN for multiple monotonic variables.
"""

from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
import torch.nn as nn


def _flatten(parameters: Sequence[torch.Tensor]) -> torch.Tensor:
    flat = [parameter.contiguous().view(-1) for parameter in parameters]
    if not flat:
        return torch.empty(0)
    return torch.cat(flat)


@lru_cache(maxsize=None)
def _cached_cc_weights(nb_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the official Clenshaw--Curtis weights and cosine nodes."""
    if nb_steps < 1:
        raise ValueError("nb_steps must be at least 1.")

    lam = np.arange(nb_steps + 1, dtype=np.float64).reshape(-1, 1)
    lam = np.cos((lam @ lam.T) * math.pi / nb_steps)
    lam[:, 0] = 0.5
    lam[:, -1] *= 0.5
    lam *= 2.0 / nb_steps

    moments = np.arange(nb_steps + 1, dtype=np.float64).reshape(-1, 1)
    moments[np.arange(1, nb_steps + 1, 2)] = 0.0
    moments = 2.0 / (1.0 - moments**2)
    moments[0] = 1.0
    moments[np.arange(1, nb_steps + 1, 2)] = 0.0

    weights = (lam.T @ moments).astype(np.float32)
    nodes = np.cos(
        np.arange(nb_steps + 1, dtype=np.float64).reshape(-1, 1)
        * math.pi
        / nb_steps
    ).astype(np.float32)
    return weights, nodes


def compute_cc_weights(nb_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Public tensor form retained for comparison with the upstream code."""
    weights, nodes = _cached_cc_weights(int(nb_steps))
    return torch.from_numpy(weights.copy()), torch.from_numpy(nodes.copy())


def _quadrature(
    x0: torch.Tensor,
    x: torch.Tensor,
    integrand: nn.Module,
    context: torch.Tensor,
    nb_steps: int,
) -> torch.Tensor:
    weights, nodes = compute_cc_weights(nb_steps)
    weights = weights.to(device=x.device, dtype=x.dtype)
    nodes = nodes.to(device=x.device, dtype=x.dtype)

    x0_steps = x0.unsqueeze(1).expand(-1, nb_steps + 1, -1)
    x_steps = x.unsqueeze(1).expand_as(x0_steps)
    context_steps = context.unsqueeze(1).expand(-1, nb_steps + 1, -1)
    nodes = nodes.unsqueeze(0).expand(x.shape[0], -1, x.shape[1])

    samples = x0_steps + (x_steps - x0_steps) * (nodes + 1.0) / 2.0
    samples = samples.contiguous().view(-1, x.shape[1])
    context_steps = context_steps.contiguous().view(-1, context.shape[1])
    values = integrand(samples, context_steps).view(x.shape[0], nb_steps + 1, -1)
    weighted = values * weights.unsqueeze(0)
    return weighted.sum(dim=1) * (x - x0) / 2.0


class ParallelNeuralIntegral(torch.autograd.Function):
    """UMNN integral with the upstream Leibniz-rule endpoint gradient."""

    @staticmethod
    def forward(ctx, x0, x, integrand, flat_params, context, nb_steps=50):
        nb_steps = int(nb_steps)
        with torch.no_grad():
            result = _quadrature(x0, x, integrand, context, nb_steps)
        ctx.integrand = integrand
        ctx.nb_steps = nb_steps
        ctx.save_for_backward(x0.detach(), x.detach(), context.detach())
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x0, x, saved_context = ctx.saved_tensors
        integrand = ctx.integrand

        # Recompute only the quadrature derivatives with respect to integrand
        # parameters and conditioning variables.  The derivative with respect
        # to the integration bounds follows the positive endpoint integrand,
        # as in the official UMNN custom backward implementation.
        with torch.enable_grad():
            context = saved_context.detach().requires_grad_(True)
            parameters = tuple(integrand.parameters())
            integral = _quadrature(
                x0.detach(), x.detach(), integrand, context, ctx.nb_steps
            )
            gradients = torch.autograd.grad(
                integral,
                parameters + (context,),
                grad_outputs=grad_output,
                allow_unused=False,
                create_graph=False,
            )
            parameter_gradient = _flatten(gradients[:-1])
            context_gradient = gradients[-1]

        with torch.no_grad():
            x_gradient = integrand(x, saved_context) * grad_output
            x0_gradient = -integrand(x0, saved_context) * grad_output

        return (
            x0_gradient,
            x_gradient,
            None,
            parameter_gradient,
            context_gradient,
            None,
        )


class IntegrandNN(nn.Module):
    """Official UMNN integrand: ReLU MLP followed by ELU + 1."""

    def __init__(self, in_features: int, hidden_layers: List[int]) -> None:
        super().__init__()
        dimensions = [int(in_features)] + [int(v) for v in hidden_layers] + [1]
        layers: List[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((x, context), dim=1)) + 1.0


class ConditionalUMNN1D(nn.Module):
    """One-dimensional conditional UMNN from the upstream implementation."""

    def __init__(
        self,
        condition_features: int,
        hidden_layers: List[int],
        nb_steps: int = 50,
    ) -> None:
        super().__init__()
        if nb_steps < 1:
            raise ValueError("nb_steps must be at least 1.")
        self.nb_steps = int(nb_steps)
        self.integrand = IntegrandNN(condition_features + 1, hidden_layers)

        dimensions = [condition_features] + [int(v) for v in hidden_layers] + [2]
        layers: List[nn.Module] = []
        for index, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU())
        self.conditioner = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError(f"ConditionalUMNN1D expects (batch, 1), got {tuple(x.shape)}.")
        factors = self.conditioner(context)
        offset = factors[:, 0:1]
        scaling = torch.exp(factors[:, 1:2])
        x0 = torch.zeros_like(x)
        integral = ParallelNeuralIntegral.apply(
            x0,
            x,
            self.integrand,
            _flatten(tuple(self.integrand.parameters())),
            context,
            self.nb_steps,
        )
        return scaling * integral + offset


# Backward-compatible name for code that imports the former one-dimensional class.
UMNN1D = ConditionalUMNN1D


class UMNNModel(nn.Module):
    """Generalized UMNN for partially monotonic scalar prediction.

    Each constrained variable is transformed by a conditional UMNN.  Their
    outputs are combined using positive exponential weights and passed through
    a second conditional UMNN.  Unconstrained features are conditioning
    variables and may therefore affect the prediction without sign constraints.
    """

    def __init__(
        self,
        input_size: int,
        monotonic_indices: List[int],
        mono_hidden_sizes: List[int],
        nonmono_hidden_sizes: Optional[List[int]] = None,
        n_integration_steps: int = 50,
        mono_activation: Optional[str] = None,
        nonmono_activation: Optional[str] = None,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.monotonic_indices = [int(index) for index in monotonic_indices]
        if not self.monotonic_indices:
            raise ValueError("UMNNModel requires at least one monotonic feature.")
        if len(set(self.monotonic_indices)) != len(self.monotonic_indices):
            raise ValueError("monotonic_indices must not contain duplicates.")
        if any(index < 0 or index >= self.input_size for index in self.monotonic_indices):
            raise ValueError("monotonic_indices contains an out-of-range feature index.")
        if output_size != 1:
            raise ValueError("The generalized UMNN implementation has one scalar output.")
        if not mono_hidden_sizes or any(int(width) <= 0 for width in mono_hidden_sizes):
            raise ValueError("mono_hidden_sizes must contain positive layer widths.")

        # These legacy arguments are accepted so old saved configurations can
        # still be loaded.  The official generalized construction conditions
        # directly on free inputs and fixes ReLU/ELU activations internally.
        _ = nonmono_hidden_sizes, mono_activation, nonmono_activation

        monotonic_set = set(self.monotonic_indices)
        nonmonotonic_indices = [
            index for index in range(self.input_size) if index not in monotonic_set
        ]
        self.register_buffer(
            "monotonic_index_tensor",
            torch.tensor(self.monotonic_indices, dtype=torch.long),
        )
        self.register_buffer(
            "nonmonotonic_index_tensor",
            torch.tensor(nonmonotonic_indices, dtype=torch.long),
        )

        # A constant one-dimensional context avoids zero-width Linear layers
        # when every feature is constrained; it has no effect on monotonicity.
        self.condition_features = max(len(nonmonotonic_indices), 1)
        hidden = [int(width) for width in mono_hidden_sizes]
        self.inner_nets = nn.ModuleList(
            [
                ConditionalUMNN1D(
                    self.condition_features, hidden, n_integration_steps
                )
                for _ in self.monotonic_indices
            ]
        )
        self.weights = nn.Parameter(torch.randn(len(self.monotonic_indices)))
        self.outer_net = ConditionalUMNN1D(
            self.condition_features, hidden, n_integration_steps
        )

    def set_steps(self, nb_steps: int) -> None:
        if nb_steps < 1:
            raise ValueError("nb_steps must be at least 1.")
        for network in self.inner_nets:
            network.nb_steps = int(nb_steps)
        self.outer_net.nb_steps = int(nb_steps)

    def _context(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonmonotonic_index_tensor.numel() == 0:
            return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        return x.index_select(1, self.nonmonotonic_index_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected x with shape (batch, {self.input_size}), got {tuple(x.shape)}."
            )
        monotonic_inputs = x.index_select(1, self.monotonic_index_tensor)
        context = self._context(x)
        inner_outputs = torch.cat(
            [
                network(monotonic_inputs[:, i : i + 1], context)
                for i, network in enumerate(self.inner_nets)
            ],
            dim=1,
        )
        inner_sum = (torch.exp(self.weights).unsqueeze(0) * inner_outputs).sum(
            dim=1, keepdim=True
        )
        return self.outer_net(inner_sum, context)
