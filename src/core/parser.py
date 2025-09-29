"""Core UI parsing functionality."""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from ..config import PARSING_CONFIDENCE_THRESHOLD


@dataclass
class UIElement:
    type: str
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


@dataclass
class ParseResult:
    elements: List[UIElement]
    
    def find(self, text_contains: str = None, element_type: str = None) -> Optional[UIElement]:
        for element in self.elements:
            if text_contains and text_contains.lower() in element.text.lower():
                return element
            if element_type and element.type == element_type:
                return element
        return None
    
    def visualize(self, img: Image.Image, show_labels: bool = True) -> Image.Image:
        from PIL import ImageDraw, ImageFont
        
        img_copy = img.copy()
        if img_copy.mode != 'RGB':
            img_copy = img_copy.convert('RGB')
        draw = ImageDraw.Draw(img_copy)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
            label_font = ImageFont.load_default()
        
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 165, 0),  # Orange
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 255, 0),  # Yellow
            (128, 0, 128),  # Purple
        ]
        
        for i, element in enumerate(self.elements):
            x1, y1, x2, y2 = element.bbox
            color = colors[i % len(colors)]
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            if show_labels:
                label = f"{i+1}. {element.type.capitalize()}"
                
                bbox = draw.textbbox((0, 0), label, font=label_font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                label_y = max(5, y1 - text_height - 5)
                
                draw.rectangle([x1, label_y, x1 + text_width + 6, label_y + text_height + 4], 
                             fill=(255, 255, 255), outline=color, width=1)
                
                draw.text((x1 + 3, label_y + 2), label, fill=color, font=label_font)
        
        return img_copy
    
    def save_analysis(self, filepath: Path, img_size: tuple = None) -> None:
        """Save detailed analysis to text file."""
        with open(filepath, 'w') as f:
            f.write("UI ELEMENT ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            if img_size:
                f.write(f"Image Size: {img_size[0]} x {img_size[1]} pixels\n")
            
            f.write(f"Total Elements Detected: {len(self.elements)}\n")
            
            buttons = [e for e in self.elements if e.type == "button"]
            text_regions = [e for e in self.elements if e.type == "text"]
            
            f.write(f"- Buttons: {len(buttons)}\n")
            f.write(f"- Text Regions: {len(text_regions)}\n\n")
            
            f.write("DETAILED ELEMENT LIST\n")
            f.write("-" * 30 + "\n\n")
            
            for i, element in enumerate(self.elements, 1):
                x1, y1, x2, y2 = element.bbox
                width = x2 - x1
                height = y2 - y1
                center_x, center_y = element.center
                
                f.write(f"Element {i}: {element.type.upper()}\n")
                f.write(f"  Position: ({x1}, {y1}) to ({x2}, {y2})\n")
                f.write(f"  Size: {width} x {height} pixels\n")
                f.write(f"  Center: ({center_x}, {center_y})\n")
                f.write(f"  Confidence: {element.confidence:.2f}\n")
                if element.text:
                    f.write(f"  Text: '{element.text}'\n")
                f.write("\n")
            
            f.write("AUTOMATION COORDINATES\n")
            f.write("-" * 30 + "\n\n")
            f.write("Click coordinates for automation:\n")
            for i, element in enumerate(self.elements, 1):
                center_x, center_y = element.center
                f.write(f"Element {i} ({element.type}): click at ({center_x}, {center_y})\n")


class UIParser:
    def __init__(self):
        self.confidence_threshold = PARSING_CONFIDENCE_THRESHOLD
    
    def parse(self, img: Image.Image) -> ParseResult:
        elements = []
        
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        button_elements = self._detect_buttons(gray)
        text_elements = self._detect_text_regions(gray)
        
        elements.extend(button_elements)
        elements.extend(text_elements)
        
        filtered_elements = [e for e in elements if e.confidence >= self.confidence_threshold]
        
        return ParseResult(elements=filtered_elements)
    
    def _detect_buttons(self, gray: np.ndarray) -> List[UIElement]:
        elements = []
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 0.5 <= aspect_ratio <= 4.0:
                    element = UIElement(
                        type="button",
                        text="",
                        bbox=(x, y, x + w, y + h),
                        confidence=0.7
                    )
                    elements.append(element)
        
        return elements
    
    def _detect_text_regions(self, gray: np.ndarray) -> List[UIElement]:
        elements = []
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(gray, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                
                if w > 20 and h > 10:
                    element = UIElement(
                        type="text",
                        text="",
                        bbox=(x, y, x + w, y + h),
                        confidence=0.6
                    )
                    elements.append(element)
        
        return elements
