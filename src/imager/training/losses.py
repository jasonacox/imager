"""Loss functions for training."""

import torch
import torch.nn as nn


class DiffusionLoss(nn.Module):
    """Diffusion training loss."""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, noise_pred: torch.Tensor, noise_target: torch.Tensor) -> torch.Tensor:
        """Compute diffusion loss.
        
        Args:
            noise_pred: Predicted noise
            noise_target: Target noise
            
        Returns:
            Loss value
        """
        return self.mse_loss(noise_pred, noise_target)
