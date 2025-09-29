"""Simple demo application for OmniParser with Granite VLM."""

from src import UIParser, GraniteVLM, ScreenCapture, setup_directories
from src.config import OUTPUT_DIR
from pathlib import Path


def main():
    setup_directories()
    
    print("Initializing UI Parser and Granite VLM...")
    parser = UIParser()
    vlm = GraniteVLM()
    capture = ScreenCapture()
    
    print("Capturing screenshot...")
    screenshot = capture.capture_active_window()
    screenshot_path = capture.save_screenshot(screenshot)
    print(f"Screenshot saved: {screenshot_path}")
    
    print("Parsing UI elements...")
    result = parser.parse(screenshot)
    print(f"Found {len(result.elements)} UI elements")
    
    for i, element in enumerate(result.elements):
        print(f"Element {i+1}: {element.type} at {element.bbox} (confidence: {element.confidence:.2f})")
    
    print("Creating visualization...")
    visualization = result.visualize(screenshot, show_labels=True)
    vis_path = OUTPUT_DIR / "parsed_ui.png"
    visualization.save(vis_path)
    print(f"Visualization saved: {vis_path}")
    
    print("Analyzing UI with Granite VLM...")
    analysis = vlm.analyze_ui(screenshot)
    print(f"VLM Analysis: {analysis}")
    
    analysis_path = OUTPUT_DIR / "vlm_analysis.txt"
    with open(analysis_path, 'w') as f:
        f.write(f"UI Analysis:\n{analysis}\n\n")
        f.write(f"Detected Elements:\n")
        for i, element in enumerate(result.elements):
            f.write(f"{i+1}. {element.type} at {element.bbox} (confidence: {element.confidence:.2f})\n")
    
    print(f"Analysis saved: {analysis_path}")
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()
