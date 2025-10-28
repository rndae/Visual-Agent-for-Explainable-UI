# Environment Setup Guide

This project requires **two separate virtual environments** due to conflicting transformers versions:

## 🔧 Environment Requirements

### 1. **OmniParser Environment** (`.venv`)
- **Purpose**: Run OmniParser v2 with Florence2 and YOLO
- **transformers version**: 4.45.2 (required for Florence2)
- **Commands**: `run_omniparser.py`

### 2. **Qwen VLM Environment** (`.venv-qwen`)  
- **Purpose**: Run Qwen3-VL for explainable UI actions
- **transformers version**: 4.57.0+ (required for Qwen3VLForConditionalGeneration)
- **Commands**: `run_vlm.sh` or `src.vlm_cli`

---

## 📦 Installation

### Step 1: Create OmniParser Environment

```bash
# Create and activate main environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install OmniParser dependencies
pip install -r requirements.txt

# Test OmniParser
python run_omniparser.py images/form-example-2.png
```

### Step 2: Create Qwen VLM Environment

```bash
# Create separate Qwen environment
python -m venv .venv-qwen
source .venv-qwen/bin/activate

# Install Qwen dependencies (latest transformers from git)
pip install git+https://github.com/huggingface/transformers
pip install torch torchvision accelerate pillow qwen-vl-utils

# OR use requirements file
pip install -r requirements-qwen.txt

# Test Qwen VLM
./run_vlm.sh images/form-example-2.png --user-data "Fill form: John Doe, 123456789"
```

---

## 🚀 Usage

### Run OmniParser Only (Use `.venv`)

```bash
source .venv/bin/activate
python run_omniparser.py images/form-example-2.png -v
```

### Run Full VLM Pipeline (Use `.venv-qwen`)

```bash
# Option 1: Use wrapper script (automatically activates correct env)
./run_vlm.sh images/form-example-2.png --user-data "Fill form with: Jose de la Rosa, 242343111, josedelarosaroja@jose.com"

# Option 2: Manually activate Qwen environment
source .venv-qwen/bin/activate
python -m src.vlm_cli images/form-example-2.png --user-data "..."

# Option 3: Use mock VLM for testing (works in either environment)
python -m src.vlm_cli images/form-example-2.png --mock-vlm
```

---

## 🎯 Quick Command Reference

| Task | Environment | Command |
|------|-------------|---------|
| **Parse UI elements** | `.venv` | `python run_omniparser.py <image>` |
| **VLM explanation** | `.venv-qwen` | `./run_vlm.sh <image> --user-data "..."` |
| **Mock VLM test** | Either | `python -m src.vlm_cli <image> --mock-vlm` |

---

## ⚠️ Troubleshooting

### Error: `cannot import name 'Qwen3VLForConditionalGeneration'`
**Solution**: You're using the wrong environment. Switch to `.venv-qwen`:
```bash
source .venv-qwen/bin/activate
```

### Error: `AttributeError: '_tie_or_clone_weights'`
**Solution**: You're using the wrong environment. Switch to `.venv`:
```bash
source .venv/bin/activate
```

### OmniParser works but VLM fails
**Solution**: Install Qwen dependencies in separate environment:
```bash
python -m venv .venv-qwen
source .venv-qwen/bin/activate
pip install -r requirements-qwen.txt
```

---

## 📝 Why Two Environments?

- **Florence2** (OmniParser) requires `transformers==4.45.2`
- **Qwen3-VL** requires `transformers>=4.57.0`
- These versions are **incompatible** - they cannot coexist in the same environment
- Solution: Separate environments, each with the correct transformers version
