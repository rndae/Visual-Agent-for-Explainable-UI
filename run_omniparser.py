"""Main entry point for OmniParser v2 UI element detection.

This script processes images using Microsoft's OmniParser v2 with YOLO icon detection
and Florence2 captioning to identify and annotate UI elements.

Usage:
    python run_omniparser.py <image_path> [options]

Example:
    python run_omniparser.py images/form-example-2.png
    python run_omniparser.py images/form-example-2.png -o custom_output
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

from src import UIParser, setup_directories
from src.config import OUTPUT_DIR


def run_omniparser(image_path: str, output_prefix: str = None, verbose: bool = False) -> bool:
    """Run OmniParser v2 on an image file.
    
    Args:
        image_path: Path to input image
        output_prefix: Optional custom prefix for output files
        verbose: Enable detailed logging
        
    Returns:
        True if successful, False otherwise
    """
    setup_directories()
    
    img_path = Path(image_path)
    
    if not img_path.exists():
        print(f"Error: Image file '{img_path}' not found")
        return False
    
    if output_prefix is None:
        output_prefix = img_path.stem
    
    if verbose:
        print(f"Loading image: {img_path}")
    
    try:
        img = Image.open(img_path)
        if verbose:
            print(f"Image size: {img.size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    if verbose:
        print("Initializing OmniParser v2...")
    
    parser = UIParser()
    
    if verbose:
        print("Parsing UI elements...")
    
    result = parser.parse(img, use_paddleocr=False)
    
    num_elements = len(result.elements)
    buttons = [e for e in result.elements if e.type in ("button", "icon")]
    text_regions = [e for e in result.elements if e.type == "text"]
    
    print(f"Found {num_elements} UI elements")
    print(f"  - {len(text_regions)} text elements")
    print(f"  - {len(buttons)} icon/button elements")
    
    if verbose and result.elements:
        print("\nDetected Elements:")
        for i, element in enumerate(result.elements, 1):
            x1, y1, x2, y2 = element.bbox
            width = x2 - x1
            height = y2 - y1
            center_x, center_y = element.center
            desc = f": {element.text[:30]}..." if element.text and len(element.text) > 30 else (f": {element.text}" if element.text else "")
            print(f"  {i}. {element.type.capitalize()}{desc}")
            print(f"     Position: ({x1}, {y1}) to ({x2}, {y2}) | Center: ({center_x}, {center_y})")
    
    print("\nGenerating visualization...")
    visualization = result.visualize(img, show_labels=True)
    vis_path = OUTPUT_DIR / f"{output_prefix}_parsed.png"
    visualization.save(vis_path, format='PNG', quality=100)
    print(f"  Saved: {vis_path}")
    
    print("Generating analysis report...")
    analysis_path = OUTPUT_DIR / f"{output_prefix}_analysis.txt"
    result.save_analysis(analysis_path, img.size)
    print(f"  Saved: {analysis_path}")
    
    print(f"\nProcessing completed successfully!")
    print(f"  Input: {img_path}")
    print(f"  Elements detected: {num_elements}")
    print(f"  Output directory: {OUTPUT_DIR}")
    
    return True


def main():
    """Command-line interface for OmniParser v2."""
    parser = argparse.ArgumentParser(
        description="Run OmniParser v2 UI element detection on an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_omniparser.py images/form-example-2.png
  python run_omniparser.py images/screenshot.png -o my_analysis -v
        """
    )
    
    parser.add_argument(
        "image",
        help="Path to input image file"
    )
    parser.add_argument(
        "-o", "--output",
        help="Custom prefix for output files (default: input filename)",
        default=None
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with detailed element information"
    )
    
    args = parser.parse_args()
    
    success = run_omniparser(args.image, args.output, args.verbose)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
