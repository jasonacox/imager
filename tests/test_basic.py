"""Basic tests for the Imager package."""

import pytest
import torch

from imager.models.text_encoder import CLIPTextEncoder
from imager.models.unet import UNet
from imager.models.diffusion import DiffusionModel
from imager.training.losses import DiffusionLoss
from imager.training.schedulers import NoiseScheduler
from imager.utils.config import Config


def test_config_loading():
    """Test configuration loading."""
    config = Config()
    assert config.model.text_encoder_name == "openai/clip-vit-base-patch32"
    assert config.training.batch_size == 4
    assert config.inference.num_inference_steps == 50


def test_text_encoder():
    """Test text encoder."""
    encoder = CLIPTextEncoder()
    
    # Test single text
    embeddings = encoder.encode("hello world")
    assert embeddings.shape == (1, 512)
    
    # Test multiple texts
    embeddings = encoder.encode(["hello", "world"])
    assert embeddings.shape == (2, 512)
    
    # Test max length
    assert encoder.get_max_length() == 77


def test_unet():
    """Test UNet model."""
    unet = UNet()
    
    # Test forward pass
    sample = torch.randn(1, 4, 64, 64)
    timestep = torch.randint(0, 1000, (1,))
    
    output = unet(sample, timestep)
    assert output.shape == sample.shape


def test_diffusion_model():
    """Test complete diffusion model."""
    text_encoder = CLIPTextEncoder()
    unet = UNet()
    model = DiffusionModel(text_encoder, unet)
    
    # Test forward pass
    noisy_latents = torch.randn(1, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (1,))
    text_inputs = "a beautiful landscape"
    
    output = model(noisy_latents, timesteps, text_inputs)
    assert output.shape == noisy_latents.shape


def test_loss_function():
    """Test diffusion loss."""
    loss_fn = DiffusionLoss()
    
    noise_pred = torch.randn(1, 4, 64, 64)
    noise_target = torch.randn(1, 4, 64, 64)
    
    loss = loss_fn(noise_pred, noise_target)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


def test_noise_scheduler():
    """Test noise scheduler."""
    scheduler = NoiseScheduler()
    
    original_samples = torch.randn(1, 4, 64, 64)
    noise = torch.randn(1, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (1,))
    
    noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
    assert noisy_samples.shape == original_samples.shape


if __name__ == "__main__":
    pytest.main([__file__])
