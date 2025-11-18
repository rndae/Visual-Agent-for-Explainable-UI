"""
OmniParser API Server
Provides UI element detection and parsing via REST API
Similar to vlm_api.py but for OmniParser functionality
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Add tools directory to path
TOOLS_PATH = os.path.join(os.path.dirname(__file__), 'tools')
if TOOLS_PATH not in sys.path:
    sys.path.insert(0, TOOLS_PATH)

# Import OmniParser tool
from omniparser_tool import get_screen_elements

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
PORT = int(os.getenv('OMNIPARSER_PORT', 5001))
HOST = os.getenv('OMNIPARSER_HOST', '0.0.0.0')

logger.info("✓ Using local OmniParser models")


def save_base64_image(base64_data: str, output_path: str) -> str:
    """Save base64 encoded image to file"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))
        image.save(output_path)
        return output_path
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")


def format_elements_as_text(elements: list) -> str:
    """Convert OmniParser elements to formatted text"""
    if not elements:
        return "No UI elements detected."
    
    lines = ["UI Element Analysis:", "=" * 70, ""]
    
    for idx, element in enumerate(elements, 1):
        elem_type = element.get("type", "unknown")
        text_content = element.get("text", "")
        bbox = element.get("bbox", [0, 0, 0, 0])
        center = element.get("center", [0, 0])
        confidence = element.get("confidence", 0.0)
        
        lines.append(f"Element {idx}:")
        lines.append(f"  Type: {elem_type}")
        if text_content:
            lines.append(f"  Text: \"{text_content}\"")
        lines.append(f"  Center Position: ({center[0]}, {center[1]})")
        lines.append(f"  Bounding Box: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}")
        lines.append(f"  Confidence: {confidence:.2f}")
        lines.append("")
    
    lines.append(f"Summary: {len(elements)} interactive elements detected")
    
    return "\n".join(lines)


@app.route('/api/parse', methods=['POST'])
def parse_interface():
    """
    Parse UI interface from image
    
    Accepts:
    - image_path: local file path
    - image_data: base64 encoded image
    - url: website URL (will take screenshot)
    
    Returns:
    - elements: list of detected UI elements
    - analysis_text: formatted text description
    - element_count: number of elements detected
    """
    try:
        data = request.json or {}
        
        # Get image from different sources
        image_path = data.get('image_path')
        image_data = data.get('image_data')
        url = data.get('url')
        
        temp_file = None
        
        # Handle URL screenshot
        if url:
            logger.info(f"Capturing screenshot of URL: {url}")
            sys.path.insert(0, TOOLS_PATH) if TOOLS_PATH not in sys.path else None
            from screenshot_tool import capture_website_screenshot
            temp_file = capture_website_screenshot(url)
            image_path = temp_file
        
        # Handle base64 image
        elif image_data:
            logger.info("Processing base64 image data")
            temp_dir = 'temp_uploads'
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, 'temp_image.png')
            save_base64_image(image_data, temp_file)
            image_path = temp_file
        
        # Validate image path
        if not image_path or not os.path.exists(image_path):
            return jsonify({
                "status": "error",
                "error": "No valid image provided. Use 'image_path', 'image_data', or 'url'"
            }), 400
        
        logger.info(f"Processing image: {image_path}")
        
        # Process with OmniParser
        elements = get_screen_elements(image_path)
        
        # Check for errors
        if elements and isinstance(elements, list) and len(elements) > 0:
            if "error" in elements[0]:
                error_msg = elements[0]["error"]
                logger.error(f"OmniParser error: {error_msg}")
                return jsonify({
                    "status": "error",
                    "error": error_msg
                }), 500
        
        # Format as text for LLM
        analysis_text = format_elements_as_text(elements)
        
        # Save analysis to file
        output_dir = 'data/outputs/omniparser'
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        analysis_path = os.path.join(output_dir, f"{base_name}_analysis.txt")
        
        with open(analysis_path, 'w') as f:
            f.write(analysis_text)
        
        logger.info(f"✓ Detected {len(elements)} UI elements")
        
        # Clean up temp file
        if temp_file and os.path.exists(temp_file) and 'temp_' in temp_file:
            try:
                os.remove(temp_file)
            except:
                pass
        
        return jsonify({
            "status": "success",
            "data": {
                "elements": elements,
                "analysis_text": analysis_text,
                "analysis_path": analysis_path,
                "element_count": len(elements),
                "image_path": image_path
            }
        })
    
    except Exception as e:
        logger.error(f"Parse error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "service": "OmniParser API",
        "version": "1.0.0",
        "mode": "local",
        "endpoints": {
            "/api/parse": "Parse UI elements from image",
            "/api/health": "Health check"
        }
    })


def print_startup_banner():
    """Print startup information"""
    print("=" * 70)
    print("🔍 OmniParser API v1.0.0 (Local Models)")
    print("=" * 70)
    print(f"Server: http://{HOST}:{PORT}")
    print()
    print("Configuration:")
    print(f"  • Mode: Local OmniParser v2")
    print(f"  • Tools Path: {TOOLS_PATH}")
    print()
    print("Endpoints:")
    print(f"  POST /api/parse         - Parse UI elements")
    print(f"  GET  /api/health        - Health check")
    print("=" * 70)
    print()


if __name__ == '__main__':
    print_startup_banner()
    logger.info(f"Starting OmniParser API on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
