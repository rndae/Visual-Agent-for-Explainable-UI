"""CLI for VLM analysis of OmniParser output.

Usage:
  python -m src.vlm_cli <image_path> [--user-data "..."]

Prerequisite: OmniParser analysis file must exist in data/outputs/
"""
from __future__ import annotations

import argparse
import os
import importlib.util
from typing import Optional
from pathlib import Path

# Load vlm_client directly without triggering src package imports
_vlm_client_path = os.path.join(os.path.dirname(__file__), "vlm_client.py")
_spec = importlib.util.spec_from_file_location("vlm_client", _vlm_client_path)
_vlm_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_vlm_client)
QwenVLMClient = _vlm_client.QwenVLMClient


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run VLM analysis on OmniParser output")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--user-data", help="User instructions/data to map into the UI", default=None)

    args = parser.parse_args(argv)

    image_path = args.image
    analysis_path = os.path.join("data", "outputs", f"{Path(image_path).stem}_analysis.txt")
    
    if not os.path.exists(analysis_path):
        print(f"Error: OmniParser analysis not found: {analysis_path}")
        print(f"Run OmniParser first: python run_omniparser.py {image_path}")
        return 2

    client = QwenVLMClient()
    out = client.run_with_omniparser(image_path, analysis_txt_path=analysis_path, user_instructions=args.user_data)
    
    print("VLM Process Complete")
    print(f"Summary: {out.get('summary', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
