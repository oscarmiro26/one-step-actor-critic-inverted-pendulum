import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple Multi-layer perceptron model.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Multi-layer perceptron initialization.

        :param input_size: Number of input features.
        :param output_size: Number of output features.
        :param dropout_rate: Dropout rate between 0 and 1. Default is 0.5.
        """
        super(MLP, self).__init__()
        self._layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the multi-layer perceptron.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        return self._layers(x)
