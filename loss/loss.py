import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeCoherenceLoss(nn.Module):
    def __init__(self, connectivity_weight: float, sparsity_weight: float, smoothness_weight: float, line_weight: float,
                 target_density: float):
        super(ShapeCoherenceLoss, self).__init__()
        self.connectivity_weight: float = connectivity_weight
        self.sparsity_weight: float = sparsity_weight
        self.smoothness_weight: float = smoothness_weight
        self.line_weight: float = line_weight
        self.target_density: float = target_density

    def get_connectivity_loss(self, x):
        # 8-connectivity kernel
        kernel = torch.tensor([[1., 1., 1.], [1., 0., 1.], [1., 1., 1.]], device=x.device).view(1, 1, 3, 3)
        neighbor_count = F.conv2d(x, kernel, padding=1)
        isolation_penalty = torch.exp(-0.5 * neighbor_count) * x
        return isolation_penalty.mean()

    def get_sparsity_loss(self, x):
        current_density = x.mean()
        return F.mse_loss(current_density, torch.tensor(self.target_density, device=x.device))

    def get_smoothness_loss(self, x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        return (dx.mean() + dy.mean()) / 2.0

    def get_line_structure_loss(self, x):
        # Line-like structures via horizontal, vertical, and diagonal edge filters
        edge_kernels = torch.stack(
            [torch.tensor([[-1., -1., -1.], [2., 2., 2.], [-1., -1., -1.]], device=x.device),  # Horizontal

             torch.tensor([[-1., 2., -1.], [-1., 2., -1.], [-1., 2., -1.]], device=x.device),  # Vertical

             torch.tensor([[2., -1., -1.], [-1., 2., -1.], [-1., -1., 2.]], device=x.device),  # Diagonal 1

             torch.tensor([[-1., -1., 2.], [-1., 2., -1.], [2., -1., -1.]], device=x.device)  # Diagonal 2
             ]).view(4, 1, 3, 3)

        # Apply edge detection and measure response to encourage line-like patterns
        edge_responses = torch.sum(torch.abs(F.conv2d(x, edge_kernels, padding=1)), dim=0)
        line_structure_loss = 1 - edge_responses.mean()  # Maximize line-like response
        return line_structure_loss

    def forward(self, x):
        connectivity_loss = self.get_connectivity_loss(x)
        sparsity_loss = self.get_sparsity_loss(x)
        smoothness_loss = self.get_smoothness_loss(x)
        line_structure_loss = self.get_line_structure_loss(x)

        total_loss = (
                self.connectivity_weight * connectivity_loss + self.sparsity_weight * sparsity_loss + self.smoothness_weight * smoothness_loss + self.line_weight * line_structure_loss)

        return total_loss
