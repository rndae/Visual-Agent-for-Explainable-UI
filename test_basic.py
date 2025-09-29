"""Basic functionality test."""

import torch
from PIL import Image
import numpy as np
from src import UIParser, setup_directories
from src.config import DEVICE


def test_basic_functionality():
    """Test basic parsing without VLM to ensure core functionality works."""
    setup_directories()
    
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    parser = UIParser()
    
    test_img = Image.new('RGB', (800, 600), color='white')
    
    result = parser.parse(test_img)
    print(f"Parsed {len(result.elements)} elements from test image")
    
    visualization = result.visualize(test_img)
    print("Visualization created successfully")
    
    print("Basic functionality test passed!")


if __name__ == "__main__":
    test_basic_functionality()
