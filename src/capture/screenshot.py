"""Screenshot capture functionality."""

import mss
import numpy as np
from PIL import Image
from pathlib import Path
from ..config import SCREENSHOT_DIR, SCREENSHOT_FORMAT


class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()
        
    def capture_active_window(self) -> Image.Image:
        monitor = self.sct.monitors[1]
        screenshot = self.sct.grab(monitor)
        
        img_array = np.array(screenshot)
        img = Image.fromarray(img_array, mode='RGB')
        
        return img
    
    def capture_region(self, x: int, y: int, width: int, height: int) -> Image.Image:
        region = {
            "top": y,
            "left": x,
            "width": width,
            "height": height
        }
        
        screenshot = self.sct.grab(region)
        img_array = np.array(screenshot)
        img = Image.fromarray(img_array, mode='RGB')
        
        return img
    
    def save_screenshot(self, img: Image.Image, filename: str = None) -> Path:
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.{SCREENSHOT_FORMAT.lower()}"
        
        filepath = SCREENSHOT_DIR / filename
        img.save(filepath, format=SCREENSHOT_FORMAT)
        
        return filepath
