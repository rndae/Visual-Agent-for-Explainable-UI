"""Main package initialization."""

from .core import UIParser, UIElement, ParseResult
from .utils import setup_directories

_all_exports = ["UIParser", "UIElement", "ParseResult", "setup_directories"]

try:
    from .capture import ScreenCapture
    _all_exports.append("ScreenCapture")
except ImportError:
    pass

try:
    from .models import GraniteVLM
    _all_exports.append("GraniteVLM")
except ImportError:
    pass

__all__ = _all_exports
