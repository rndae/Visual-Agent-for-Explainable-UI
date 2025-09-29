"""Lightweight test for sample image without VLM."""

from PIL import Image
from pathlib import Path
from src import UIParser, setup_directories
from src.config import OUTPUT_DIR

def test_sample_light():
    setup_directories()
    
    sample_path = Path("images/sample1.png")
    img = Image.open(sample_path)
    
    print(f"Testing OmniParser on {sample_path}")
    print(f"Image dimensions: {img.size}")
    
    parser = UIParser()
    result = parser.parse(img)
    
    print(f"\nDetected {len(result.elements)} UI elements:")
    
    buttons = [e for e in result.elements if e.type == "button"]
    text_regions = [e for e in result.elements if e.type == "text"]
    
    print(f"- {len(buttons)} buttons")
    print(f"- {len(text_regions)} text regions")
    
    print("\nButton locations:")
    for i, btn in enumerate(buttons, 1):
        x1, y1, x2, y2 = btn.bbox
        center_x, center_y = btn.center
        print(f"  {i}. Button at center ({center_x}, {center_y}) - size: {x2-x1}x{y2-y1}")
    
    if text_regions:
        print("\nText region locations:")
        for i, txt in enumerate(text_regions, 1):
            x1, y1, x2, y2 = txt.bbox
            center_x, center_y = txt.center
            print(f"  {i}. Text at center ({center_x}, {center_y}) - size: {x2-x1}x{y2-y1}")
    
    print("\nCreating high-quality visualization...")
    visualization = result.visualize(img, show_labels=True)
    vis_path = OUTPUT_DIR / "sample1_parsed_improved.png"
    visualization.save(vis_path, format='PNG', quality=100)
    print(f"Improved visualization saved to: {vis_path}")
    
    print("Saving detailed text analysis...")
    analysis_path = OUTPUT_DIR / "sample1_analysis.txt"
    result.save_analysis(analysis_path, img.size)
    print(f"Detailed analysis saved to: {analysis_path}")
    
    test_element_finding(result)

def test_element_finding(result):
    print("\nTesting element finding capabilities:")
    
    if result.elements:
        first_button = result.find(element_type="button")
        if first_button:
            print(f"Found first button at: {first_button.center}")
        
        first_text = result.find(element_type="text")
        if first_text:
            print(f"Found first text region at: {first_text.center}")
        
        print(f"Total elements available for interaction: {len(result.elements)}")

if __name__ == "__main__":
    test_sample_light()
