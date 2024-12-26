import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Distribution

from models.twoheadedmlp import TwoHeadedMLP


class MLPMultivariateGaussian(TwoHeadedMLP):
    """
    A multilayer perceptron with two output heads to represent a multivariate Gaussian distribution.
    This type of neural network can be used to output a (multivariate) Gaussian distribution,
    in this case one output head is for the mean and the other are elements of the covariance matrix.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the MLP with two output heads.

        :param input_size: Number of input features.
        :param output_size: Number of output features.
        """
        super(MLPMultivariateGaussian, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(128, output_size)  # Linear layer for mean
        # https://en.wikipedia.org/wiki/Cholesky_decomposition
        self.log_diag_chol_head = torch.nn.Linear(128,
                                                  output_size) # Linear layer for log diagonal of Cholesky decomposition
                                                               # The log is for numerical stability.

        self.double()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MLP.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Tuple containing mean tensor of shape (batch_size, output_size)
                 and log diagonal of Cholesky decomposition tensor of shape (batch_size, output_size).
        """
        features = self.layers(x)
        mean = self.mean_head(features)
        log_diag_chol = self.log_diag_chol_head(features)
        return mean, log_diag_chol

    def predict(self, x: torch.Tensor) -> Distribution:
        """
        Predict a MultivariateNormal distribution given an input tensor.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: MultivariateNormal distribution.

        The forward pass gives a mean and log of the diagonal elements of the Cholesky decomposition
        of the covariance matrix.
        For a full prediction, we need to construct the Multivariate Gaussian distribution,
        and for that, we need to exponentiate the log of the diagonal elements and then reconstruct
        the covariance matrix using the Cholesky decomposition.

        For more information, see the Wikipedia page on [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix).
        """
        mean, log_diag_chol = self(x)

        # Construct the covariance matrix using Cholesky decomposition
        cov_matrix = torch.diag_embed(torch.exp(log_diag_chol))

        return MultivariateNormal(mean, cov_matrix)
