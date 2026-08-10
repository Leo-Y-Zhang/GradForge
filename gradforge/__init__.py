"""GradForge: a from-scratch reverse-mode autograd engine and a tiny GPT
that prove their own gradients. See docs/DESIGN.md for the architecture."""
from .gradcheck import gradcheck
from .tensor import Tensor

__version__ = "0.1.0"
__all__ = ["Tensor", "gradcheck", "__version__"]
