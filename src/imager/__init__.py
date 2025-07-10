"""
Imager: An open-source text-to-image generator built with diffusion models and transformers.

This package provides a modular, research-focused framework for generating high-quality 
images from text prompts using diffusion probabilistic models and transformer architectures.
"""

__version__ = "0.1.0"
__author__ = "Imager Team"
__email__ = "contact@imager.ai"

from .models import *
from .training import *
from .inference import *
from .utils import *
