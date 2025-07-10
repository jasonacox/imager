"""Data preprocessing utilities."""

from typing import Any, List, Union

import torch
import torchvision.transforms as transforms
from PIL import Image


class ImagePreprocessor:
    """Preprocessor for images."""
    
    def __init__(self, size: int = 512, normalize: bool = True):
        transform_list = [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
            
        self.transform = transforms.Compose(transform_list)
        
    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Preprocess image."""
        if isinstance(image, torch.Tensor):
            return image
        return self.transform(image)


class TextPreprocessor:
    """Preprocessor for text."""
    
    def __init__(self, max_length: int = 77):
        self.max_length = max_length
        
    def __call__(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """Preprocess text."""
        if isinstance(text, str):
            text = [text]
            
        # Placeholder - will implement actual tokenization later
        return {
            "input_ids": torch.randint(0, 1000, (len(text), self.max_length)),
            "attention_mask": torch.ones(len(text), self.max_length),
        }
