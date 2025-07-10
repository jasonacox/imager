"""Diffusion model implementation."""

from typing import Optional, Union

import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .unet import UNet


class DiffusionModel(nn.Module):
    """Complete diffusion model combining text encoder and UNet."""
    
    def __init__(
        self,
        text_encoder: TextEncoder,
        unet: UNet,
        scheduler: Optional[object] = None,
    ):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        
    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_inputs: Union[str, list],
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            noisy_latents: Noisy latent inputs
            timesteps: Diffusion timesteps
            text_inputs: Text prompts
            
        Returns:
            Predicted noise
        """
        # Encode text
        encoder_hidden_states = self.text_encoder.encode(text_inputs)
        
        # Predict noise
        noise_pred = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        )
        
        return noise_pred
