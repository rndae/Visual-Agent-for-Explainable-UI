"""Core UI parsing functionality using OmniParser v2."""

import sys
import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import torch

project_root = Path(__file__).parent.parent.parent
omniparser_path = project_root / "weights" / "OmniParser"
sys.path.insert(0, str(omniparser_path))

from util.utils import get_yolo_model, get_caption_model_processor, check_ocr_box, get_som_labeled_img

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
        self._initialize_models()
    
    def _initialize_models(self):
        weights_path = Path(__file__).parent.parent.parent / "weights" / "OmniParser" / "weights"
        
        self.yolo_model = get_yolo_model(model_path=str(weights_path / "icon_detect" / "model.pt"))
        self.caption_model_processor = get_caption_model_processor(
            model_name="florence2", 
            model_name_or_path=str(weights_path / "icon_caption_florence")
        )
    
    def parse(self, img: Image.Image, box_threshold: float = 0.05, iou_threshold: float = 0.1, use_paddleocr: bool = False, imgsz: int = 640) -> ParseResult:
        elements = []
        
        img_array = np.array(img)
        
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            img, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        box_overlay_ratio = img.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            img, 
            self.yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=False, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=self.caption_model_processor, 
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz,
        )
        
        for idx, element_dict in enumerate(parsed_content_list):
            element_type = element_dict.get('type', 'button')
            element_content = element_dict.get('content', '')
            bbox_ratio = element_dict.get('bbox', [0, 0, 0, 0])
            
            w, h = img.size
            x1 = int(bbox_ratio[0] * w)
            y1 = int(bbox_ratio[1] * h)
            x2 = int(bbox_ratio[2] * w)
            y2 = int(bbox_ratio[3] * h)
            
            element = UIElement(
                type=element_type,
                text=element_content if isinstance(element_content, str) else "",
                bbox=(x1, y1, x2, y2),
                confidence=0.7
            )
            elements.append(element)
        
        filtered_elements = [e for e in elements if e.confidence >= self.confidence_threshold]
        
        return ParseResult(elements=filtered_elements)
