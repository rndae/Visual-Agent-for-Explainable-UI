#!/bin/bash
# Wrapper script to run VLM analysis with the correct Python environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN_VENV="$SCRIPT_DIR/.venv-qwen"

if [ ! -d "$QWEN_VENV" ]; then
    echo "Error: Qwen environment not found at $QWEN_VENV"
    echo "Please run: python -m venv .venv-qwen && source .venv-qwen/bin/activate && pip install -r requirements-qwen.txt"
    exit 1
fi

# Activate Qwen environment and run the VLM analysis
source "$QWEN_VENV/bin/activate"
python run_vlm_analysis.py "$@"
