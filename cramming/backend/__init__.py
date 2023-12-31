"""This module implements interfaces to the various backends."""

from .prepare_backend import load_backend
from .torch_default import _load_optimizer

__all__ = ["load_backend"]
