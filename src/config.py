"""Configuration module with global constants."""

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCREENSHOT_DIR = DATA_DIR / "screenshots"
OUTPUT_DIR = DATA_DIR / "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRANITE_MODEL_PATH = "ibm-granite/granite-vision-3.3-2b"

BATCH_SIZE = 1
NUM_WORKERS = 4
MAX_NEW_TOKENS = 100

SCREENSHOT_FORMAT = "PNG"
SCREENSHOT_QUALITY = 95

DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 64

PARSING_CONFIDENCE_THRESHOLD = 0.5

DIRECTORIES = [
    SCREENSHOT_DIR,
    OUTPUT_DIR,
]
