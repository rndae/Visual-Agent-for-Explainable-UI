import sys
import os
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any

# Add project root to path to import OmniParser
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import UIParser

def get_screen_elements(image_path: str) -> List[Dict[str, Any]]:
    """
    Analyzes a screenshot to identify interactable UI elements, text, and icons using local OmniParser.
    
    Args:
        image_path (str): The local file path to the screenshot (e.g., "screenshot.png").

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a UI element
                    containing 'text', 'type', 'bbox' (in absolute pixels), 'center', and 'confidence'.
    """
    try:
        # 1. Load image
        if not os.path.exists(image_path):
            return [{"error": f"File not found: {image_path}"}]
        
        img = Image.open(image_path)
        width, height = img.size

        # 2. Initialize local OmniParser
        parser = UIParser()
        
        # 3. Parse UI elements (use_paddleocr=False for faster processing)
        result = parser.parse(img, use_paddleocr=False)
        
        # 4. Convert to agent-friendly format
        cleaned_elements = []
        
        for element in result.elements:
            # UIElement has: type, text, bbox, center, confidence
            cleaned_elements.append({
                "text": element.text or "",
                "type": element.type,  # 'text', 'icon', 'button'
                "bbox": list(element.bbox),  # [x1, y1, x2, y2] in pixels
                "center": list(element.center),  # [x, y]
                "confidence": element.confidence
            })
        
        return cleaned_elements

    except Exception as e:
        # Return the error in a struct so the agent knows what happened
        return [{"error": f"OmniParser analysis failed: {str(e)}"}]

# ------------------------------------------------------------------
# EXAMPLE AGENT USAGE
# ------------------------------------------------------------------
# if __name__ == "__main__":
#     image_file = "screenshot.png"  # Make sure this file exists locally
    
#     print(f"🔎 Analyzing '{image_file}'...")
    
#     # Call the tool
#     elements = get_screen_elements(image_file)
    
#     # Check for execution errors (e.g., file not found, API down)
#     if elements and "error" in elements[0]:
#         print(f"❌ Error: {elements[0]['error']}")
#     else:
#         print(f"✅ Success! Found {len(elements)} elements.\n")
        
#         # PRINT FULL OUTPUT
#         # We use json.dumps with indent=2 to make it readable in the terminal
#         print(json.dumps(elements, indent=2, ensure_ascii=False))


# Example of importing and using in another script

# from omniparser_tool import get_screen_elements

# # In your agent loop
# screenshot_path = agent.take_screenshot()
# ui_elements = get_screen_elements(screenshot_path)

# # Pass 'ui_elements' into the prompt context
# prompt = f"Here is what is on the screen: {json.dumps(ui_elements)}"