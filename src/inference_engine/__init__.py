"""A small, readable inference engine for learning model and GPU execution."""

from .model import ModelConfig, TinyDecoder
from .generation import generate

__all__ = ["ModelConfig", "TinyDecoder", "generate"]
