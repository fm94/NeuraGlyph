import torch
import torch.nn as nn


class Decrypter(nn.Module):
    def __init__(self, num_classes: int, image_size: int):
        """
        Initialize the Decrypter model, a convolutional neural network for classifying images and getting letter labels.

        Args:
            num_classes (int): Number of output classes.
            image_size (int): Input image size (must be a multiple of 8).
        """
        super(Decrypter, self).__init__()
        self.num_classes: int = num_classes
        self.image_size: int = image_size

        # Ensure the image_size is a multiple of 8 to match the downsampling layers
        assert image_size % 8 == 0, "image_size must be a multiple of 8 for the current architecture."

        # Define the convolutional layers
        self.conv_net = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Output: same size
                                      nn.ReLU(), nn.MaxPool2d(2),  # Halves the spatial dimensions

                                      nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: same size
                                      nn.ReLU(), nn.MaxPool2d(2),  # Halves the spatial dimensions

                                      nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: same size
                                      nn.ReLU(), nn.MaxPool2d(2)  # Halves the spatial dimensions
                                      )

        # Initialize fully connected layers
        self.fc_net: nn.Sequential | None = None
        self._initialize_fc_layer()

    def _initialize_fc_layer(self) -> None:
        """
        Initialize the fully connected layers by determining the flattened size after convolution.
        """
        # Create a dummy input to determine the output size after convolution layers
        dummy_input = torch.zeros(1, 1, self.image_size, self.image_size)
        conv_output = self.conv_net(dummy_input)  # Forward pass to get output size
        flattened_size = conv_output.numel()  # Total number of features after flattening

        # Define the fully connected layers based on the calculated flattened size
        self.fc_net = nn.Sequential(nn.Flatten(), nn.Linear(flattened_size, 256), nn.ReLU(),
                                    nn.Linear(256, self.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Decrypter network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, image_size, image_size).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, num_classes), representing class scores.
        """
        x = self.conv_net(x)
        x = self.fc_net(x)
        return x
