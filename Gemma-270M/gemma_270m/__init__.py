"""
Gemma-270M: A lightweight implementation of the Gemma 270M parameter model.

This package provides a clean, modular implementation suitable for training,
fine-tuning, and inference with the Gemma-270M model architecture.
"""

__version__ = "0.1.0"

from .model import GemmaModel, GemmaConfig, get_default_config

__all__ = [
    "GemmaModel",
    "GemmaConfig",
    "get_default_config",
]
