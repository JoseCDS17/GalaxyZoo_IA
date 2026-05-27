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

def product(integers: tuple[int, ...]) -> int:
    result = 1
    for element in integers:
        result *= element
    return result


class DoubleConvolutionBlock(nn.Module): 
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
        self.bn1 = nn.BatchNorm2d(
            hidden_channels
        )
        self.conv2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=kernel_size, padding=padding_size
        )
        self.bn2 = nn.BatchNorm2d(
            out_channels
        )
        self.activation = (
            get_activation(activation) if activation is not None else nn.ReLU()
        )
        if add_residual and maintain_size:
            if in_channels != out_channels:
                self.residual_projection = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1
                )
            else:
                self.residual_projection = nn.Identity()
        else:
            self.residual_projection = None

    def forward(self, x: torch.Tensor):
        out1 = self.activation(
            self.bn1(self.conv1(x))
        )
        out2 = self.bn2(self.conv2(out1))

        #residual connection, used in modern networks.
        if self.residual_projection is not None:
            return out2 + self.residual_projection(x)
        
        return self.activation(out2)
    

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
            nn.AdaptiveAvgPool2d((1,1)), #for reducing our parameters
            nn.Flatten(),
            nn.Linear(cnn_output_channels,final_hidden_channels),
            get_activation(activation),
            nn.Linear(final_hidden_channels, out_channels),
        ]

        if task == "regression":
            mlp_layers.append(nn.Sigmoid())

        self.mlp_model = nn.Sequential(*mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_model(self.cnn_blocks(x))

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
    
        for i in range(n_blocks):
            # We would get an error if we try to use a kernel size that is larger than our image size
            if any(dimension < 3 for dimension in shape):
                raise ValueError(
                    "Dimensions of input tensor must be at least 3 for convolutional layers."
                )
            
            out_channels = hidden_channels * (2**i)

            cnn_layers.append(
                DoubleConvolutionBlock(current_channels, out_channels)
            ) 
            current_channels = out_channels
            cnn_layers.append(nn.MaxPool2d(kernel_size=(2, 2))) 
            shape = tuple(dimension // 2 for dimension in shape)

        self.cnn_blocks = nn.Sequential(*cnn_layers)

        return shape, current_channels
