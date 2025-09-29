"""Utility helper functions."""

from pathlib import Path
from ..config import DIRECTORIES


def setup_directories():
    """Create necessary directories if they don't exist."""
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def validate_image_path(image_path: Path) -> bool:
    """Validate if image path exists and is a valid image file."""
    if not image_path.exists():
        return False
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    return image_path.suffix.lower() in valid_extensions
