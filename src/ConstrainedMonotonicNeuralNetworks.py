"""PyTorch port of the constrained monotonic dense architectures.

The implementation follows the public ``monotonic-nn`` MonoDense layer and
its Type-1/Type-2 model builders.  In particular, hidden units are split
between convex, concave, and saturated activations; these are fixed ratios,
not trainable activation weights.
"""

from functools import lru_cache
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import init_weights


Activation = Optional[Union[str, Callable[[torch.Tensor], torch.Tensor]]]


class MonoDense(nn.Module):
    """Monotonic counterpart of a fully connected layer.

    ``monotonicity_indicator`` is +1 for increasing inputs, -1 for
    decreasing inputs, and 0 for unconstrained inputs.  The activation split
    is the PyTorch equivalent of the official MonoDense ``apply_activations``
    routine.
    """

    def __init__(
        self,
        in_features: int,
        units: int,
        activation: Activation = None,
        monotonicity_indicator: Union[int, List[int], torch.Tensor] = 1,
        is_convex: bool = False,
        is_concave: bool = False,
        activation_weights: Tuple[float, float, float] = (7.0, 7.0, 2.0),
        init_method: Literal[
            "xavier_uniform", "xavier_normal", "kaiming_uniform",
            "kaiming_normal", "he_uniform", "he_normal",
            "truncated_normal",
        ] = "xavier_uniform",
    ) -> None:
        super().__init__()
        if is_convex and is_concave:
            raise ValueError("A MonoDense layer cannot be both convex and concave.")
        if len(activation_weights) != 3 or any(v < 0 for v in activation_weights):
            raise ValueError("activation_weights must contain three non-negative values.")
        if sum(activation_weights) <= 0:
            raise ValueError("At least one activation weight must be positive.")

        self.in_features = int(in_features)
        self.units = int(units)
        self.org_activation = activation
        self.is_convex = bool(is_convex)
        self.is_concave = bool(is_concave)
        self.activation_weights = tuple(float(v) for v in activation_weights)
        self.init_method = init_method

        indicator = self.get_monotonicity_indicator(
            monotonicity_indicator, self.in_features, self.units
        )
        self.register_buffer("monotonicity_indicator", indicator)

        self.weight = nn.Parameter(torch.empty(self.units, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.units))
        self.reset_parameters()

        (
            self.convex_activation,
            self.concave_activation,
            self.saturated_activation,
        ) = self.get_activation_functions(self.org_activation)

    def reset_parameters(self) -> None:
        init_weights(self.weight, method=self.init_method)
        init_weights(self.bias, method="zeros")

    @staticmethod
    def get_monotonicity_indicator(
        monotonicity_indicator: Union[int, List[int], torch.Tensor],
        in_features: int,
        units: int,
    ) -> torch.Tensor:
        indicator = torch.as_tensor(monotonicity_indicator, dtype=torch.float32)
        if indicator.ndim == 0:
            indicator = indicator.repeat(in_features)
        if indicator.ndim == 1:
            if indicator.numel() not in (1, in_features):
                raise ValueError(
                    "monotonicity_indicator must be scalar or have one value per input."
                )
            indicator = indicator.reshape(-1, 1)
        elif indicator.ndim != 2:
            raise ValueError("monotonicity_indicator must have rank at most 2.")

        try:
            indicator = indicator.expand(in_features, units).t().contiguous()
        except RuntimeError as exc:
            raise ValueError(
                f"Cannot broadcast monotonicity_indicator to ({in_features}, {units})."
            ) from exc

        if not torch.all((indicator == -1) | (indicator == 0) | (indicator == 1)):
            raise ValueError("monotonicity_indicator values must be -1, 0, or 1.")
        return indicator

    @staticmethod
    @lru_cache(maxsize=None)
    def get_activation_functions(activation: Activation):
        if callable(activation):
            convex = activation
        else:
            name = activation.lower() if isinstance(activation, str) else activation
            activations = {
                "relu": F.relu,
                "elu": F.elu,
                "selu": F.selu,
                "gelu": F.gelu,
                "tanh": torch.tanh,
                "sigmoid": torch.sigmoid,
                None: None,
                "linear": None,
            }
            if name not in activations:
                raise ValueError(f"Unsupported activation: {activation}")
            convex = activations[name]

        if convex is None:
            return None, None, None

        def concave(x: torch.Tensor) -> torch.Tensor:
            return -convex(-x)

        return convex, concave, MonoDense.get_saturated_activation(convex, concave)

    @staticmethod
    def get_saturated_activation(
        convex_activation: Callable[[torch.Tensor], torch.Tensor],
        concave_activation: Callable[[torch.Tensor], torch.Tensor],
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def saturated(x: torch.Tensor) -> torch.Tensor:
            cc = convex_activation(torch.ones_like(x))
            return torch.where(
                x <= 0,
                convex_activation(x + 1.0) - cc,
                concave_activation(x - 1.0) + cc,
            )

        return saturated

    def apply_monotonicity_indicator_to_kernel(
        self, kernel: torch.Tensor
    ) -> torch.Tensor:
        abs_kernel = torch.abs(kernel)
        result = torch.where(self.monotonicity_indicator == 1, abs_kernel, kernel)
        return torch.where(self.monotonicity_indicator == -1, -abs_kernel, result)

    def _activation_sizes(self) -> Tuple[int, int, int]:
        if self.is_convex:
            ratios = (1.0, 0.0, 0.0)
        elif self.is_concave:
            ratios = (0.0, 1.0, 0.0)
        else:
            total = sum(self.activation_weights)
            ratios = tuple(v / total for v in self.activation_weights)

        n_convex = round(ratios[0] * self.units)
        n_concave = round(ratios[1] * self.units)
        n_saturated = self.units - n_convex - n_concave
        if n_saturated < 0:
            # Guard against an unusual rounding edge case while preserving the
            # official allocation rule for ordinary layer widths.
            n_concave += n_saturated
            n_saturated = 0
        return n_convex, n_concave, n_saturated

    def apply_activations(self, h: torch.Tensor) -> torch.Tensor:
        if self.convex_activation is None:
            return h

        sizes = self._activation_sizes()
        pieces = torch.split(h, sizes, dim=-1)
        return torch.cat(
            (
                self.convex_activation(pieces[0]),
                self.concave_activation(pieces[1]),
                self.saturated_activation(pieces[2]),
            ),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.apply_monotonicity_indicator_to_kernel(self.weight)
        return self.apply_activations(F.linear(x, weight, self.bias))


class _DenseFeatureExtractor(nn.Module):
    """Ordinary per-feature extractor used for Type-2 free variables."""

    def __init__(self, units: int, activation: Activation, init_method: str) -> None:
        super().__init__()
        self.linear = nn.Linear(1, units)
        init_weights(self.linear.weight, method=init_method)
        init_weights(self.linear.bias, method="zeros")
        self.activation = MonoDense.get_activation_functions(activation)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        return h if self.activation is None else self.activation(h)


class ConstrainedMonotonicNeuralNetwork(nn.Module):
    """Type-1 or Type-2 constrained monotonic neural network."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        device: torch.device,
        activation: str = "elu",
        monotonicity_indicator: Optional[List[int]] = None,
        final_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        init_method: str = "xavier_uniform",
        architecture_type: str = "type1",
        activation_weights: Tuple[float, float, float] = (7.0, 7.0, 2.0),
    ) -> None:
        super().__init__()
        if not hidden_sizes or any(int(width) <= 0 for width in hidden_sizes):
            raise ValueError("hidden_sizes must contain at least one positive width.")
        if architecture_type not in {"type1", "type2"}:
            raise ValueError("architecture_type must be 'type1' or 'type2'.")

        self.input_size = int(input_size)
        self.hidden_sizes = [int(width) for width in hidden_sizes]
        self.output_size = int(output_size)
        self.activation = activation
        self.final_activation = final_activation
        self.architecture_type = architecture_type
        self.init_method = init_method
        self.activation_weights = activation_weights

        if monotonicity_indicator is None:
            monotonicity_indicator = [1] * self.input_size
        indicator = torch.as_tensor(monotonicity_indicator, dtype=torch.float32)
        if indicator.ndim != 1 or indicator.numel() != self.input_size:
            raise ValueError(
                "monotonicity_indicator must contain one value per input feature."
            )
        if not torch.all((indicator == -1) | (indicator == 0) | (indicator == 1)):
            raise ValueError("monotonicity_indicator values must be -1, 0, or 1.")
        self.register_buffer("monotonicity_indicator", indicator)

        self.feature_extractors = nn.ModuleList()
        self.network = (
            self._build_type1() if architecture_type == "type1" else self._build_type2()
        )
        self.to(device)

    def _mono_dense(
        self,
        in_features: int,
        units: int,
        activation: Activation,
        indicator: Union[int, List[int], torch.Tensor],
    ) -> MonoDense:
        return MonoDense(
            in_features=in_features,
            units=units,
            activation=activation,
            monotonicity_indicator=indicator,
            activation_weights=self.activation_weights,
            init_method=self.init_method,
        )

    def _build_common_block(
        self, in_features: int, first_indicator: Union[int, List[int], torch.Tensor]
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        previous = in_features
        for layer_index, width in enumerate(self.hidden_sizes):
            indicator = first_indicator if layer_index == 0 else 1
            layers.append(self._mono_dense(previous, width, self.activation, indicator))
            previous = width
        layers.append(self._mono_dense(previous, self.output_size, None, 1))
        return layers

    def _build_type1(self) -> nn.ModuleList:
        return self._build_common_block(self.input_size, self.monotonicity_indicator)

    def _build_type2(self) -> nn.ModuleList:
        input_units = max(self.hidden_sizes[0] // 4, 1)
        common_indicator: List[int] = []
        for value in self.monotonicity_indicator.tolist():
            if value == 0:
                extractor: nn.Module = _DenseFeatureExtractor(
                    input_units, self.activation, self.init_method
                )
            else:
                extractor = self._mono_dense(
                    1, input_units, self.activation, int(value)
                )
            self.feature_extractors.append(extractor)
            common_indicator.extend([abs(int(value))] * input_units)

        return self._build_common_block(
            self.input_size * input_units, common_indicator
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected x with shape (batch, {self.input_size}), got {tuple(x.shape)}."
            )
        if self.architecture_type == "type2":
            x = torch.cat(
                [layer(x[:, i : i + 1]) for i, layer in enumerate(self.feature_extractors)],
                dim=1,
            )
        for layer in self.network:
            x = layer(x)
        return self.final_activation(x) if self.final_activation is not None else x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
