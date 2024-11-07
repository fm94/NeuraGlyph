import torch
import torch.nn as nn


class Encryper(nn.Module):
    def __init__(self, noise_dim: int, num_classes: int, temperature: float, image_size: int):
        """
        Initialize the Encryper model, an upsampling neural network for generating images from noise and conditions.

        Args:
            noise_dim (int): Dimension of the noise input vector.
            num_classes (int): Number of classes for conditional generation.
            temperature (float): Temperature parameter for controlling sigmoid scaling.
            image_size (int): Desired output image size (must be a multiple of 8).
        """
        super(Encryper, self).__init__()
        self.noise_dim = noise_dim
        self.temperature = temperature
        self.output_height, self.output_width = image_size, image_size

        # Ensure image_size is valid for upsampling
        assert image_size % 8 == 0, "image_size must be a multiple of 8 for the current architecture."

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, 64)

        # Calculate the initial dense layer size based on output dimensions
        initial_dense_height = self.output_height // 8
        initial_dense_width = self.output_width // 8

        # Ensure the initial dense layer size is valid
        assert initial_dense_height > 0 and initial_dense_width > 0, "Output size must be at least 8x8 for the current architecture."

        # Initial dense layer to reshape input
        self.initial_dense = nn.Linear(noise_dim + 64, 128 * initial_dense_height * initial_dense_width)

        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                                         # 2x upsampling
                                         nn.BatchNorm2d(64), nn.LeakyReLU(0.2),

                                         nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                         # 4x upsampling
                                         nn.BatchNorm2d(32), nn.LeakyReLU(0.2),

                                         nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                                         # 8x upsampling
                                         nn.BatchNorm2d(16), nn.LeakyReLU(0.2),

                                         nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1)  # Final output layer
                                         )

    def forward(self, z: torch.Tensor, labels: torch.Tensor, binary: bool = False) -> torch.Tensor:
        """
        Forward pass through the Encryper network.

        Args:
            z (torch.Tensor): Latent input vector of shape (batch_size, noise_dim).
            labels (torch.Tensor): Class labels tensor of shape (batch_size,).
            binary (bool): Whether to return binary output (default: False).

        Returns:
            torch.Tensor: Output image tensor of shape (batch_size, 1, output_height, output_width).
        """
        # Concatenate latent vector with class embedding
        class_embedding = self.class_embedding(labels)
        x = torch.cat([z, class_embedding], dim=1)

        # Initial dense and reshape
        x = self.initial_dense(x)
        initial_shape = (128, self.output_height // 8, self.output_width // 8)
        x = x.view(-1, *initial_shape)

        # Apply conv blocks for upsampling
        x = self.conv_blocks(x)

        # Apply temperature-scaled sigmoid
        x = torch.sigmoid(x / self.temperature)

        if binary:
            x = (x > 0.5).float()

        return x
