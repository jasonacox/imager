"""Configuration management for Imager."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # Text encoder config
    text_encoder_name: str = "openai/clip-vit-base-patch32"
    text_encoder_max_length: int = 77
    text_embedding_dim: int = 512
    
    # UNet config
    unet_in_channels: int = 4
    unet_out_channels: int = 4
    unet_down_block_types: List[str] = field(default_factory=lambda: [
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"
    ])
    unet_up_block_types: List[str] = field(default_factory=lambda: [
        "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
    ])
    unet_block_out_channels: List[int] = field(default_factory=lambda: [320, 640, 1280, 1280])
    unet_layers_per_block: int = 2
    unet_attention_head_dim: int = 8
    unet_cross_attention_dim: int = 768
    
    # Diffusion config
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    
    # VAE config (for latent diffusion)
    vae_name: str = "stabilityai/sd-vae-ft-mse"
    latent_channels: int = 4
    latent_scale_factor: float = 0.18215


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Data
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 4
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Scheduler
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    
    # Checkpointing
    save_every_n_epochs: int = 5
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Logging
    log_every_n_steps: int = 100
    use_wandb: bool = False
    wandb_project: str = "imager"
    wandb_run_name: Optional[str] = None
    
    # Validation
    val_every_n_epochs: int = 1
    num_val_samples: int = 4
    val_prompts: List[str] = field(default_factory=lambda: [
        "a beautiful landscape",
        "a cat sitting on a table",
        "abstract art",
        "a futuristic city"
    ])


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    
    # Generation
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    
    # Sampling
    sampler: str = "ddim"  # "ddpm", "ddim", "dpm_solver"
    eta: float = 0.0
    
    # Output
    output_dir: str = "./outputs"
    save_intermediate: bool = False
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    mixed_precision: bool = True


@dataclass
class Config:
    """Main configuration class."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Global settings
    seed: int = 42
    project_name: str = "imager"
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = OmegaConf.structured(self)
        with open(path, "w") as f:
            OmegaConf.save(config_dict, f)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return OmegaConf.structured(self)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file or return default."""
    if config_path is None:
        return Config()
    
    if isinstance(config_path, str):
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return Config.from_yaml(config_path)
