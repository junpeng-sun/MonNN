from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def uniformPWL_mono_reg(
    model: nn.Module,
    x: torch.Tensor,
    monotonic_indices: List[int],
    b: float = 0.2,
    n_points: int = 1024,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> torch.Tensor:
    """Monte-Carlo uniform monotonicity regularizer from Liu et al. (2020).

    A fresh set of points is drawn on every call.  Experiments scale the
    outer-training domain to [0, 1], so these bounds cover the complete domain
    rather than only the current mini-batch.
    """
    if not monotonic_indices:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    if n_points < 1:
        raise ValueError("n_points must be positive")
    if upper_bound <= lower_bound:
        raise ValueError("upper_bound must exceed lower_bound")

    reg_points = torch.empty(
        (n_points, x.shape[1]), device=x.device, dtype=x.dtype
    ).uniform_(lower_bound, upper_bound).requires_grad_(True)
    predictions = model(reg_points)
    grads = torch.autograd.grad(
        predictions.sum(), reg_points, create_graph=True
    )[0][:, monotonic_indices]
    penalty = torch.relu(float(b) - grads).square()
    return penalty.sum(dim=1).mean()


def _load_gurobi():
    """Import the optional verifier only when certification is requested."""
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:  # pragma: no cover - depends on local license setup
        raise RuntimeError(
            "Exact certification requires gurobipy and a usable Gurobi license."
        ) from exc
    return gp, GRB


def _linear_interval(
    weight: np.ndarray,
    bias: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positive = np.maximum(weight, 0.0)
    negative = np.minimum(weight, 0.0)
    return (
        positive @ lower + negative @ upper + bias,
        positive @ upper + negative @ lower + bias,
    )


def _certify_single_derivative(
    layers: Sequence[nn.Module],
    feature_index: int,
    direction: int,
    input_lower: np.ndarray,
    input_upper: np.ndarray,
    tolerance: float,
    time_limit: Optional[float],
) -> tuple[bool, float, str]:
    """Minimize one signed input derivative of a scalar ReLU MLP exactly."""
    gp, GRB = _load_gurobi()
    problem = gp.Model(f"monotonicity_feature_{feature_index}")
    problem.Params.OutputFlag = 0
    if time_limit is not None:
        problem.Params.TimeLimit = float(time_limit)

    values = [
        problem.addVar(lb=float(lo), ub=float(hi), name=f"x_{i}")
        for i, (lo, hi) in enumerate(zip(input_lower, input_upper))
    ]
    derivatives = [gp.LinExpr(1.0 if i == feature_index else 0.0)
                   for i in range(len(values))]
    value_lower = input_lower.copy()
    value_upper = input_upper.copy()
    deriv_lower = np.eye(len(values), dtype=np.float64)[feature_index]
    deriv_upper = deriv_lower.copy()

    for layer_index, layer in enumerate(layers):
        if isinstance(layer, nn.Identity):
            continue

        if isinstance(layer, nn.Linear):
            weight = layer.weight.detach().cpu().numpy().astype(np.float64)
            if layer.bias is None:
                bias = np.zeros(weight.shape[0], dtype=np.float64)
            else:
                bias = layer.bias.detach().cpu().numpy().astype(np.float64)

            pre_lower, pre_upper = _linear_interval(
                weight, bias, value_lower, value_upper
            )
            zero_bias = np.zeros(weight.shape[0], dtype=np.float64)
            dpre_lower, dpre_upper = _linear_interval(
                weight, zero_bias, deriv_lower, deriv_upper
            )

            pre_values = []
            pre_derivatives = []
            for j in range(weight.shape[0]):
                value_var = problem.addVar(
                    lb=float(pre_lower[j]), ub=float(pre_upper[j]),
                    name=f"pre_{layer_index}_{j}",
                )
                problem.addConstr(
                    value_var == gp.quicksum(
                        float(weight[j, k]) * values[k]
                        for k in range(weight.shape[1])
                    ) + float(bias[j])
                )
                derivative_var = problem.addVar(
                    lb=float(dpre_lower[j]), ub=float(dpre_upper[j]),
                    name=f"dpre_{layer_index}_{j}",
                )
                problem.addConstr(
                    derivative_var == gp.quicksum(
                        float(weight[j, k]) * derivatives[k]
                        for k in range(weight.shape[1])
                    )
                )
                pre_values.append(value_var)
                pre_derivatives.append(derivative_var)

            values = pre_values
            derivatives = pre_derivatives
            value_lower, value_upper = pre_lower, pre_upper
            deriv_lower, deriv_upper = dpre_lower, dpre_upper
            continue

        if isinstance(layer, nn.ReLU):
            relu_values = []
            relu_derivatives = []
            next_dlower = np.minimum(0.0, deriv_lower)
            next_dupper = np.maximum(0.0, deriv_upper)

            for j, (pre_value, pre_derivative) in enumerate(
                zip(values, derivatives)
            ):
                lower = float(value_lower[j])
                upper = float(value_upper[j])
                d_lower = float(deriv_lower[j])
                d_upper = float(deriv_upper[j])

                if upper <= 0.0:
                    relu_values.append(gp.LinExpr(0.0))
                    relu_derivatives.append(gp.LinExpr(0.0))
                    next_dlower[j] = 0.0
                    next_dupper[j] = 0.0
                    continue
                if lower >= 0.0:
                    relu_values.append(pre_value)
                    relu_derivatives.append(pre_derivative)
                    next_dlower[j] = d_lower
                    next_dupper[j] = d_upper
                    continue

                active = problem.addVar(vtype=GRB.BINARY,
                                        name=f"active_{layer_index}_{j}")
                relu_value = problem.addVar(
                    lb=0.0, ub=max(0.0, upper),
                    name=f"relu_{layer_index}_{j}",
                )
                problem.addConstr(relu_value >= pre_value)
                problem.addConstr(relu_value <= pre_value - lower * (1 - active))
                problem.addConstr(relu_value <= upper * active)

                relu_derivative = problem.addVar(
                    lb=float(next_dlower[j]), ub=float(next_dupper[j]),
                    name=f"drelu_{layer_index}_{j}",
                )
                # Exact linearization of drelu = active * dpre.
                problem.addConstr(relu_derivative >= d_lower * active)
                problem.addConstr(relu_derivative <= d_upper * active)
                problem.addConstr(
                    relu_derivative >= pre_derivative - d_upper * (1 - active)
                )
                problem.addConstr(
                    relu_derivative <= pre_derivative - d_lower * (1 - active)
                )
                relu_values.append(relu_value)
                relu_derivatives.append(relu_derivative)

            values = relu_values
            derivatives = relu_derivatives
            value_lower = np.maximum(0.0, value_lower)
            value_upper = np.maximum(0.0, value_upper)
            deriv_lower, deriv_upper = next_dlower, next_dupper
            continue

        raise TypeError(
            "Certification supports nn.Linear, nn.ReLU, and nn.Identity only; "
            f"got {type(layer).__name__}."
        )

    if len(derivatives) != 1:
        raise ValueError("Certification currently requires a scalar network output.")

    problem.setObjective(float(direction) * derivatives[0], GRB.MINIMIZE)
    problem.optimize()
    if problem.Status == GRB.OPTIMAL:
        minimum = float(problem.ObjVal)
        return minimum >= -tolerance, minimum, "optimal"
    if problem.Status == GRB.TIME_LIMIT:
        return False, float("nan"), "time_limit"
    return False, float("nan"), f"solver_status_{problem.Status}"


def certify_monotonicity(
    model: 'CertifiedMonotonicNetwork',
    monotonic_indices: Optional[Sequence[int]] = None,
    directions: Optional[Sequence[int]] = None,
    input_lower: float | Sequence[float] = 0.0,
    input_upper: float | Sequence[float] = 1.0,
    tolerance: float = 1e-8,
    time_limit: Optional[float] = None,
    return_details: bool = False,
):
    """Certify coordinatewise monotonicity of a bounded scalar ReLU MLP.

    Unlike the previous adjacent-layer helper, this MILP propagates both the
    activation state and the derivative through the complete Sequential model.
    A timeout or non-optimal solver status is reported as *not certified*, not
    as a proof of non-monotonicity.
    """
    layers = list(model.main_network.children())
    first_linear = next((layer for layer in layers if isinstance(layer, nn.Linear)), None)
    if first_linear is None:
        raise ValueError("The model contains no linear layer.")
    input_size = first_linear.in_features

    if monotonic_indices is None:
        monotonic_indices = list(range(model.n_monotonic_features))
    monotonic_indices = [int(index) for index in monotonic_indices]
    if directions is None:
        directions = [1] * len(monotonic_indices)
    if len(directions) != len(monotonic_indices) or any(d not in (-1, 1) for d in directions):
        raise ValueError("directions must contain one +1/-1 value per monotonic feature")

    lower = np.broadcast_to(np.asarray(input_lower, dtype=np.float64), (input_size,)).copy()
    upper = np.broadcast_to(np.asarray(input_upper, dtype=np.float64), (input_size,)).copy()
    if np.any(upper <= lower):
        raise ValueError("Every input upper bound must exceed its lower bound.")

    details = {}
    certified = True
    for feature_index, direction in zip(monotonic_indices, directions):
        if not 0 <= feature_index < input_size:
            raise IndexError(f"Monotonic feature index out of range: {feature_index}")
        feature_ok, minimum, status = _certify_single_derivative(
            layers, feature_index, int(direction), lower, upper,
            tolerance, time_limit,
        )
        details[feature_index] = {
            "certified": feature_ok,
            "minimum_signed_derivative": minimum,
            "status": status,
        }
        certified = certified and feature_ok

    return (certified, details) if return_details else certified


def certify_grad_with_gurobi(
    first_layer: nn.Linear,
    second_layer: nn.Linear,
    mono_feature_num: int,
    direction: Optional[Sequence[int]] = None,
):
    """Backward-compatible exact certification for a two-layer ReLU block."""
    block = CertifiedMonotonicNetwork(
        [first_layer, nn.ReLU(), second_layer], mono_feature_num
    )
    return certify_monotonicity(block, directions=direction)


# CertifiedMonotonicNetwork
class CertifiedMonotonicNetwork(nn.Module):
    def __init__(self, layers, n_monotonic_features):
        super(CertifiedMonotonicNetwork, self).__init__()
        self.main_network = nn.Sequential(*layers)
        self.n_monotonic_features = n_monotonic_features

    def forward(self, x):
        return self.main_network(x)
