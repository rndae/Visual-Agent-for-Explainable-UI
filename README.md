# Visual Agent for Explainable UI Actions

Implementation of UI parsing using Microsoft's OmniParser v2 with Qwen3-VL integration for explainable UI automation. The system detects UI elements and provides natural language explanations for suggested actions.

## Features

- **OmniParser v2 Integration**: Real Microsoft OmniParser with YOLO icon detection + Florence2 captioning
- **Qwen3-VL Integration**: Vision-language model for UI summarization and action mapping
- **Element Detection**: Identifies buttons, text fields, and interactive UI components
- **Explainable Actions**: Maps user instructions to specific UI elements with coordinates
- **High-quality Visualization**: Color-coded element annotations with labels
- **Command-line Interface**: Process any image with optional VLM analysis
- **Optional Granite VLM Support**: Alternative VLM implementation included

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended for VLM features)

### Important: Two Separate Environments Required

This project requires **two separate virtual environments** due to conflicting transformers versions:
- **OmniParser** (Florence2) needs `transformers==4.45.2`
- **Qwen3-VL** needs `transformers>=4.57.0`

See [SETUP.md](SETUP.md) for detailed installation instructions.

### Quick Setup

**1. OmniParser Environment (main functionality):**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Qwen VLM Environment (for explainable actions):**
```bash
python -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install -r requirements-qwen.txt
```

## Usage

### Run OmniParser (Use `.venv` environment)

```bash
source .venv/bin/activate  # Activate OmniParser environment
python run_omniparser.py images/form-example-2.png
```

### Run VLM Pipeline (Use `.venv-qwen` environment)

```bash
# Use wrapper script (automatically activates correct environment)
./run_vlm.sh images/form-example-2.png --user-data "Fill form with: Jose de la Rosa, 242343111, josedelarosaroja@jose.com, 2444 pine st, Seal Beach, CA, 90740"

# Or manually with Qwen environment
source .venv-qwen/bin/activate
python run_vlm_analysis.py images/form-example-2.png --user-data "..."
```

### Process any image:

```bash
# Run OmniParser only (use .venv)
source .venv/bin/activate
python run_omniparser.py images/form-example-2.png

# Run VLM for action generation (automatically uses .venv-qwen)
./run_vlm.sh images/form-example-2.png --command "Fill form with: Jose de la Rosa, 242343111, josedelarosaroja@jose.com, 2444 pine st, Seal Beach, CA, 90740"

# VLM without command (just analysis)
./run_vlm.sh images/form-example-2.png

# With verbose output showing all detected elements
python run_omniparser.py images/form-example-2.png -v

# With custom output name
python run_omniparser.py images/screenshot.png -o my_analysis

# Show all options
python run_omniparser.py --help
```

### Output Files

Each run generates:
- `{filename}_parsed.png` - Annotated image with numbered, color-coded UI elements
- `{filename}_analysis.txt` - Detailed OmniParser analysis with element descriptions and coordinates
- `{filename}_vlm_explanation.txt` - VLM-generated summary, action suggestions, and user intent mapping (when using VLM)

### Sample Images

Test with the included sample images:
```bash
python run_omniparser.py images/sample1.png
python run_omniparser.py images/sample2.png
python run_omniparser.py images/form-example-2.png -v
```

## What It Detects

OmniParser v2 with Florence2 model identifies:
- **Interactive Elements** - Buttons, links, input fields
- **Text Content** - Labels, headings, form field text
- **UI Components** - Checkboxes, dropdowns, icons
- **Precise Coordinates** - Bounding boxes and center points for automation
- **Semantic Captions** - Natural language descriptions of UI elements

Qwen3-VL provides:
- **UI Summaries** - Concise descriptions of the overall interface
- **Action Suggestions** - Recommended interactions based on visual analysis
- **User Intent Mapping** - Links user instructions to specific UI elements

## Granite VLM Support

This codebase includes optional support for IBM Granite Vision Language Model integration. Granite VLM and ScreenCapture are optional dependencies - the core functionality (OmniParser + Qwen3-VL) works independently.

## Requirements

See `requirements.txt` for complete dependency list. Key dependencies:
- PyTorch with CUDA support
- transformers==4.45.2 (locked for Florence2 compatibility)
- ultralytics==8.3.70 (YOLO model)
- EasyOCR for text detection
- Pillow for image processing
- supervision==0.18.0 (computer vision utilities)

Optional dependencies for Granite VLM:
- vllm (for Granite model inference)
- mss (for screen capture)

## Output Example

```
UI ELEMENT ANALYSIS REPORT
Generated: 2024-01-15 14:23:45
Image Size: 1054x875 pixels
Total Elements Detected: 28

TEXT ELEMENTS (7 found):
═══════════════════════════════════════════

Element 1: text
  Description: "Customers"
  Position: (150, 89) to (239, 110)
  Size: 89 x 21 pixels
  Center: (194, 99) - Click here for automation
  Confidence: N/A

ICON ELEMENTS (21 found):
═══════════════════════════════════════════

Element 8: icon
  Description: "A blue circular button with a white plus sign"
  Position: (989, 168) to (1023, 202)
  Size: 34 x 34 pixels
  Center: (1006, 185) - Click here for automation
  Confidence: 0.89
```

## Project Structure

```
├── src/                 # Core implementation
├── images/             # Sample test images
├── data/outputs/       # Generated analysis files
├── test_image_cli.py   # Command-line interface
└── requirements.txt    # Dependencies
```
