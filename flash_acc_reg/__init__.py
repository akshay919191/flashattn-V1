__version__ = "0.1.0"

try:
    from . import _C
except ImportError:
    from ._jit import build
    _C = build()

from .ops import flash_attn

__all__ = ["flash_attn"]