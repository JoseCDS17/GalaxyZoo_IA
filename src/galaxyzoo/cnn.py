from typing import Literal

from torch import nn
import torch

def get_activation(
    activation: Literal["relu", "sigmoid", "tanh"] = "relu",
):
    """Constructs an activation function based on the provided input."""
    if activation == "relu":
        return nn.ReLU()
    if activation == "sigmoid":
        return nn.Sigmoid()
    if activation == "tanh":
        return nn.Tanh()

    raise ValueError(f"Unknown activation function: {activation}")

# We will here show another benefit of structuring your Neural Network in a modular way.
# First we will make a component which we can reuse in the network.


def product(integers: tuple[int, ...]) -> int:
    result = 1
    for element in integers:
        result *= element
    return result


class DoubleConvolutionBlock(nn.Module):  # Remember to always inherit from nn.Module
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | None = None,
        activation: Literal["relu", "sigmoid", "tanh"] | None = None,
        kernel_size: int = 3,
        maintain_size: bool = True,
        add_residual: bool = True,
    ):
        """Creates a block with two convolutional layers.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            hidden_channels: The number of hidden channels. If None, it is set to out_channels.
            activation: The activation function to use. If None, it is set to ReLU.
            kernel_size: Size of the convolutional kernel. Defaults to 3.
            maintain_size: If True, the output size will be the same as the input size. Defaults to True.
            add_residual: If True and maintain_size is True and in_channels == out_channels, a residual connection is added.
        """
        # Always call the super constructor
        super().__init__()

        # Compute needed values
        # Adding this padding size, means that our convolutional layers preserve the image shape.
        padding_size = kernel_size // 2 if maintain_size else 0
        hidden_channels = (
            hidden_channels if hidden_channels is not None else out_channels
        )

        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=kernel_size, padding=padding_size
        )
        self.conv2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=kernel_size, padding=padding_size
        )
        self.activation = (
            get_activation(activation) if activation is not None else nn.ReLU()
        )
        self.add_residual = (
            add_residual and (in_channels == out_channels) and maintain_size
        )

    def forward(self, x: torch.Tensor):
        out1 = self.activation(self.conv1(x))
        out2 = self.activation(self.conv2(out1))

        # Another benefit of classes is that we can add more logic to the forward pass.
        # Here we used what is known as a residual connection (skip-connection),
        # which are essential for training large modern networks.
        if self.add_residual:
            return out2 + x
        return out2


# When creating convolutional neural networks, it is a good idea to keep track of the shape of the data as it passes through the network.
# This website: https://asiltureli.github.io/Convolution-Layer-Calculator/ for calculating the shape of the data after each layer.
# In this neural network we will use a kernel size of 3x3 and a padding of 1 to keep the size of the image the same.
# This means it is only the down-pooling layers which change the shape of the data.


# Now we can use this block to create a CNN
class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_shape: tuple[int, int],
        hidden_channels: int = 16,
        final_hidden_channels: int = 64,
        activation: Literal["relu", "sigmoid", "tanh"] = "relu",
        n_blocks: int = 3,
        task: Literal["classification", "regression"] = "classification"
    ):
        super().__init__()

        cnn_output_shape, cnn_output_channels = self._build_cnn(
            hidden_channels, in_channels, input_shape, n_blocks
        )

        # After our CNN blocks we will be flattening
        mlp_layers =  [
            nn.Flatten(),
            nn.Linear(
                product(cnn_output_shape) * cnn_output_channels,
                final_hidden_channels,
            ),
            get_activation(activation),
            nn.Linear(final_hidden_channels, out_channels),
        ]

        if task == "regression":
            mlp_layers.append(nn.Sigmoid())

        self.mlp_model = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_model(self.cnn_blocks(x))

    # In Python, it is customary to prefix "private" methods with a single underscore.
    # This means, that this function should not be called from outside the class.
    def _build_cnn(
        self,
        hidden_channels: int,
        in_channels: int,
        input_shape: tuple[int, int],
        n_blocks: int,
    ) -> tuple[tuple[int, int], int]:
        cnn_layers = []
        current_channels = in_channels
        shape = input_shape
        # Nothing prevents you from manually adding layers to nn.Sequential,
        # but if you want the depths to be configurable it is easier to use a loop to add the layers.
        # We first add all our layers to a list, and then we use python unpacking (* operator) to insert them into the nn.Sequential.
        for _ in range(n_blocks):
            # We would get an error if we try to use a kernel size that is larger than our image size,
            # so we would rather catch that error now instead of having to debug it later :).
            if any(dimension < 3 for dimension in shape):
                raise ValueError(
                    "Dimensions of input tensor must be at least 3 for convolutional layers."
                )
            cnn_layers.append(
                DoubleConvolutionBlock(current_channels, hidden_channels)
            )  # Preserves shape
            current_channels = hidden_channels
            cnn_layers.append(nn.AvgPool2d(kernel_size=(2, 2)))  # Halves shape
            shape = tuple(dimension // 2 for dimension in shape)

        # The CNN components of our network. Designed to learn meaningful features.
        self.cnn_blocks = nn.Sequential(*cnn_layers)

        # Try to print out the layers of the CNN components using
        # m = CNN(some_parameters)
        # print(m.cnn_blocks)
        # Do you understand the
        return shape, current_channels
