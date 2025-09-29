"""Command-line UI parser for any image."""

import sys
import argparse
from PIL import Image
from pathlib import Path
from src import UIParser, setup_directories
from src.config import OUTPUT_DIR

def parse_image(image_path, output_prefix=None):
    """Parse UI elements from an image file."""
    setup_directories()
    
    img_path = Path(image_path)
    
    if not img_path.exists():
        print(f"Error: Image file '{img_path}' not found")
        return False
    
    if output_prefix is None:
        output_prefix = img_path.stem
    
    print(f"Loading image: {img_path}")
    try:
        img = Image.open(img_path)
        print(f"Image size: {img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    print("Initializing UI Parser...")
    parser = UIParser()
    
    print("Parsing UI elements...")
    result = parser.parse(img)
    print(f"Found {len(result.elements)} UI elements")
    
    buttons = [e for e in result.elements if e.type == "button"]
    text_regions = [e for e in result.elements if e.type == "text"]
    
    print(f"- {len(buttons)} buttons")
    print(f"- {len(text_regions)} text regions")
    
    if result.elements:
        print("\nDetected Elements:")
        for i, element in enumerate(result.elements, 1):
            x1, y1, x2, y2 = element.bbox
            width = x2 - x1
            height = y2 - y1
            center_x, center_y = element.center
            print(f"  {i}. {element.type.capitalize()} at center ({center_x}, {center_y}) - size: {width}x{height}")
    
    print("\nCreating high-quality visualization...")
    visualization = result.visualize(img, show_labels=True)
    vis_path = OUTPUT_DIR / f"{output_prefix}_parsed.png"
    visualization.save(vis_path, format='PNG', quality=100)
    print(f"Visualization saved to: {vis_path}")
    
    print("Saving detailed text analysis...")
    analysis_path = OUTPUT_DIR / f"{output_prefix}_analysis.txt"
    result.save_analysis(analysis_path, img.size)
    print(f"Analysis saved to: {analysis_path}")
    
    print(f"\nSummary:")
    print(f"- Input: {img_path}")
    print(f"- Elements found: {len(result.elements)}")
    print(f"- Visualization: {vis_path}")
    print(f"- Analysis: {analysis_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Parse UI elements from image files")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("-o", "--output", help="Output prefix for generated files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Processing image: {args.image}")
        if args.output:
            print(f"Output prefix: {args.output}")
    
    success = parse_image(args.image, args.output)
    
    if success:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
