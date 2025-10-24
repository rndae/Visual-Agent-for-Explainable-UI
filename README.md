# Visual Agent for Explainable UI with OmniParserv2

Implementation of UI parsing using computer vision for detecting and analyzing UI elements in screenshots.

## Features

- UI element detection (buttons, text regions)
- High-quality visualization with color-coded elements
- Detailed text analysis with automation coordinates
- Command-line interface for any image processing
- Support for Granite VLM integration (code included)

## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (recommended for VLM features)

### Setup

1. Clone the repository and navigate to the project directory

2. Create a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Process any image with OmniParser v2:

```bash
# Basic usage
python test_image_cli.py path/to/your/image.png

# With custom output name
python test_image_cli.py images/sample1.png -o my_analysis

# Verbose output
python test_image_cli.py images/sample2.png -v

# Show all options
python test_image_cli.py --help
```

### Output Files

Each run generates:
- `{filename}_parsed.png` - Annotated image with numbered, color-coded UI elements
- `{filename}_analysis.txt` - Detailed analysis with click coordinates for automation

### Sample Images

Test with the included sample images:
```bash
python test_image_cli.py images/sample1.png
python test_image_cli.py images/sample2.png
```

## What It Detects

- **Buttons** - Interactive UI elements
- **Text regions** - Text areas in the interface
- **Bounding boxes** - Precise coordinates for each element
- **Click coordinates** - Center points for automation

## Granite VLM Support

This codebase includes support for IBM Granite Vision Language Model integration for intelligent UI analysis. The VLM can provide natural language descriptions of UI elements and suggest actions.

## Requirements

See `requirements.txt` for complete dependency list. Key dependencies:
- PyTorch with CUDA support
- OpenCV for computer vision
- Pillow for image processing
- MSS for screenshot capture

## Output Example

```
Element 1: BUTTON
  Position: (1079, 911) to (1169, 957)
  Size: 90 x 46 pixels
  Center: (1124, 934)
  Confidence: 0.70
```

## Project Structure

```
├── src/                 # Core implementation
├── images/             # Sample test images
├── data/outputs/       # Generated analysis files
├── test_image_cli.py   # Command-line interface
└── requirements.txt    # Dependencies
```
