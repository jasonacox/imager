"""Visualization utilities."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Union
from PIL import Image


def save_image(
    image: Union[torch.Tensor, np.ndarray, Image.Image],
    path: Union[str, Path],
    normalize: bool = True,
) -> None:
    """Save an image to disk.
    
    Args:
        image: Image to save
        path: Output path
        normalize: Whether to normalize the image
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # CHW format
            image = np.transpose(image, (1, 2, 0))
            
        if normalize and image.max() > 1.0:
            image = image / 255.0
            
        if image.shape[-1] == 1:  # Grayscale
            image = image.squeeze(-1)
            
        # Convert to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        
    if isinstance(image, Image.Image):
        image.save(path)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def plot_training_curves(
    losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plot training curves.
    
    Args:
        losses: Training losses
        val_losses: Validation losses (optional)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(losses, label="Training Loss", linewidth=2)
    if val_losses:
        plt.plot(val_losses, label="Validation Loss", linewidth=2)
        
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
        
    plt.close()


def create_image_grid(
    images: List[Union[torch.Tensor, np.ndarray, Image.Image]],
    nrow: int = 4,
    padding: int = 2,
    normalize: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Create a grid of images.
    
    Args:
        images: List of images
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize images
        save_path: Path to save the grid
        
    Returns:
        Image grid as numpy array
    """
    if not images:
        raise ValueError("No images provided")
        
    # Convert all images to numpy arrays
    processed_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
            
        if isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[0] in [1, 3, 4]:  # CHW format
                img = np.transpose(img, (1, 2, 0))
                
        processed_images.append(img)
    
    # Get image dimensions
    h, w = processed_images[0].shape[:2]
    channels = processed_images[0].shape[2] if processed_images[0].ndim == 3 else 1
    
    # Calculate grid dimensions
    ncol = (len(images) + nrow - 1) // nrow
    
    # Create grid
    if channels == 1:
        grid = np.full((ncol * h + (ncol - 1) * padding, nrow * w + (nrow - 1) * padding), 255, dtype=np.uint8)
    else:
        grid = np.full((ncol * h + (ncol - 1) * padding, nrow * w + (nrow - 1) * padding, channels), 255, dtype=np.uint8)
    
    for i, img in enumerate(processed_images):
        row = i // nrow
        col = i % nrow
        
        y_start = row * (h + padding)
        y_end = y_start + h
        x_start = col * (w + padding)
        x_end = x_start + w
        
        if normalize and img.max() > 1.0:
            img = img / 255.0
            
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
            
        grid[y_start:y_end, x_start:x_end] = img
    
    if save_path:
        save_image(grid, save_path, normalize=False)
        
    return grid
