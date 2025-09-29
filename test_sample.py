"""Test OmniParser on sample image."""

from PIL import Image
from pathlib import Path
from src import UIParser, GraniteVLM, setup_directories
from src.config import OUTPUT_DIR

def test_sample_image():
    setup_directories()
    
    sample_path = Path("images/sample1.png")
    
    if not sample_path.exists():
        print(f"Error: {sample_path} not found")
        return
    
    print(f"Loading sample image: {sample_path}")
    img = Image.open(sample_path)
    print(f"Image size: {img.size}")
    
    print("Initializing UI Parser...")
    parser = UIParser()
    
    print("Parsing UI elements...")
    result = parser.parse(img)
    print(f"Found {len(result.elements)} UI elements")
    
    print("\nDetected Elements:")
    for i, element in enumerate(result.elements, 1):
        x1, y1, x2, y2 = element.bbox
        width = x2 - x1
        height = y2 - y1
        print(f"{i}. {element.type.capitalize()} at ({x1}, {y1}) - {width}x{height} (confidence: {element.confidence:.2f})")
    
    print("\nCreating visualization...")
    visualization = result.visualize(img, show_labels=True)
    vis_path = OUTPUT_DIR / "sample1_parsed.png"
    visualization.save(vis_path)
    print(f"Visualization saved: {vis_path}")
    
    print("\nInitializing Granite VLM...")
    try:
        vlm = GraniteVLM()
        
        print("Analyzing UI with VLM...")
        analysis = vlm.analyze_ui(img, "Describe all the UI elements you can see in this interface and their purposes.")
        print(f"\nVLM Analysis:\n{analysis}")
        
        analysis_path = OUTPUT_DIR / "sample1_analysis.txt"
        with open(analysis_path, 'w') as f:
            f.write(f"UI Analysis for sample1.png\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Image size: {img.size}\n")
            f.write(f"Elements detected: {len(result.elements)}\n\n")
            f.write("Detected Elements:\n")
            for i, element in enumerate(result.elements, 1):
                x1, y1, x2, y2 = element.bbox
                width = x2 - x1
                height = y2 - y1
                f.write(f"{i}. {element.type.capitalize()} at ({x1}, {y1}) - {width}x{height} (confidence: {element.confidence:.2f})\n")
            f.write(f"\nVLM Analysis:\n{analysis}\n")
        
        print(f"Analysis saved: {analysis_path}")
        
    except Exception as e:
        print(f"VLM analysis failed: {e}")
        print("Continuing with vision-only parsing...")
    
    print("\nTesting element finding...")
    if result.elements:
        sample_element = result.elements[0]
        print(f"First element center: {sample_element.center}")
    
    print("Sample image test completed!")

if __name__ == "__main__":
    test_sample_image()
