from functools import lru_cache
import torch
import torch.nn as nn
from pmlayer.torch.layers import HLattice
from typing import List, Optional, Literal
from src.MLP import StandardMLP


class HLLNetwork(nn.Module):
    """
    Hybrid Lattice Layer (HLL) Network implementation using pmlayer and StandardMLP.

    This class provides a flexible HLL network with customizable lattice sizes,
    increasing dimensions, and MLP architecture.

    Attributes:
        dim (int): Total number of input dimensions.
        lattice_sizes (List[int]): Sizes of each lattice dimension.
        increasing (List[int]): Indices of input dimensions that should be increasing.
        mlp_neurons (List[int]): Sizes of hidden layers in the MLP.
        activation (nn.Module): Activation function for the MLP.
        dropout_rate (float): Dropout rate for the MLP.
        output_activation (nn.Module): Activation function for the output layer.
        init_method (str): Weight initialization method for the MLP.
        model (HLattice): The underlying HLattice model.
    """

    def __init__(
            self,
            dim: int,
            lattice_sizes: List[int],
            increasing: List[int],
            mlp_neurons: List[int],
            device: torch.device,
            activation: Optional[nn.Module] = None,
            dropout_rate: float = 0.0,
            output_activation: Optional[nn.Module] = None,
            output_link: Literal['identity', 'logit'] = 'identity',
            init_method: Literal[
                'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'truncated_normal'] = 'xavier_uniform'
    ):
        """
        Initialize the HLLNetwork.

        Args:
            dim (int): Total number of input dimensions.
            lattice_sizes (List[int]): Size of the lattices
            increasing (List[int]): Indices of input dimensions that should be increasing.
            mlp_neurons (List[int]): Sizes of hidden layers in the MLP.
            activation (nn.Module): Activation function for the MLP.
            dropout_rate (float): Dropout rate for the MLP.
            device (torch.device): Device
            init_method (str): Weight initialization method for the MLP.
        """
        super(HLLNetwork, self).__init__()
        self.input_size = dim
        self.device = device
        self.lattice_sizes = lattice_sizes
        self.increasing = increasing
        self.mlp_neurons = mlp_neurons
        self.activation = nn.ReLU() if activation is None else activation
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.output_link = output_link
        self.init_method = init_method
        self.model = self._build_model()

    @lru_cache(maxsize=None)
    def _build_model(self) -> HLattice:
        input_len = self.input_size - len(self.increasing)
        output_len = torch.prod(torch.tensor(self.lattice_sizes)).item()

        ann = StandardMLP(
            input_size=input_len,
            hidden_sizes=self.mlp_neurons,
            output_size=output_len,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            init_method=self.init_method,
            output_activation=self.output_activation
        )
        return HLattice(self.input_size, torch.tensor(self.lattice_sizes), self.increasing, ann)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        output = self.model(x)
        if self.output_link == 'identity':
            return output
        if self.output_link == 'logit':
            # HLattice with a sigmoid auxiliary network produces values in
            # [0, 1].  A logit link gives the common experiments an unbounded
            # regression output / classification logit while preserving the
            # lattice's monotonicity because logit is strictly increasing.
            eps = torch.finfo(output.dtype).eps
            return torch.logit(output.clamp(min=eps, max=1.0 - eps))
        raise ValueError(f"Unsupported HLL output link: {self.output_link}")


    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the network.

        Returns:
            int: The total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
