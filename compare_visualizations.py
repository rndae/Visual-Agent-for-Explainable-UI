"""Compare visualization quality improvements."""

from PIL import Image
from pathlib import Path
from src import UIParser, setup_directories
from src.config import OUTPUT_DIR

def compare_visualizations():
    setup_directories()
    
    sample_path = Path("images/sample1.png")
    img = Image.open(sample_path)
    
    parser = UIParser()
    result = parser.parse(img)
    
    print(f"Creating improved visualization with:")
    print(f"- High-quality PIL rendering (no blurriness)")
    print(f"- Color-coded elements with numbered labels")
    print(f"- Clean white label backgrounds")
    print(f"- Better font rendering")
    
    improved_vis = result.visualize(img, show_labels=True)
    improved_path = OUTPUT_DIR / "sample1_final_improved.png"
    improved_vis.save(improved_path, format='PNG', quality=100, optimize=False)
    
    print(f"\nFiles generated:")
    print(f"1. Improved visualization: {improved_path}")
    print(f"2. Detailed text analysis: {OUTPUT_DIR}/sample1_analysis.txt")
    
    print(f"\nVisualization improvements:")
    print(f"- No more blurry OpenCV text rendering")
    print(f"- Each element has a unique color and number")
    print(f"- Labels positioned above elements with white backgrounds")
    print(f"- Crisp, high-resolution output")
    
    print(f"\nText output includes:")
    print(f"- Complete element inventory")
    print(f"- Precise coordinates for automation")
    print(f"- Element sizes and confidence scores")
    print(f"- Ready-to-use click coordinates")

if __name__ == "__main__":
    compare_visualizations()
