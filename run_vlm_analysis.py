#!/usr/bin/env python3
"""Standalone VLM analysis script for OmniParser output.

Usage:
  python run_vlm_analysis.py <image_path> [--user-data "..."]

Prerequisite: OmniParser analysis file must exist in data/outputs/
"""
import argparse
import os
import importlib.util
from pathlib import Path

# Load vlm_client directly without package imports
vlm_client_path = os.path.join("src", "vlm_client.py")
spec = importlib.util.spec_from_file_location("vlm_client", vlm_client_path)
vlm_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlm_client)
QwenVLMClient = vlm_client.QwenVLMClient


def main():
    parser = argparse.ArgumentParser(description="Run VLM analysis on OmniParser output")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--command", help="User command/task to execute on the UI (e.g., 'Fill form with: John Doe, 123456789, john@email.com')", default=None)

    args = parser.parse_args()

    image_path = args.image
    analysis_path = os.path.join("data", "outputs", f"{Path(image_path).stem}_analysis.txt")
    
    if not os.path.exists(analysis_path):
        print(f"Error: OmniParser analysis not found: {analysis_path}")
        print(f"Run OmniParser first: python run_omniparser.py {image_path}")
        return 2

    print("Loading Qwen3-VL model...")
    client = QwenVLMClient()
    
    print("Running VLM analysis...")
    out = client.run_with_omniparser(image_path, analysis_txt_path=analysis_path, user_command=args.command)
    
    print("\n" + "="*50)
    print("VLM Analysis Complete")
    print("="*50)
    print(f"\nSummary: {out.get('summary', '')}")
    print(f"\nFull output saved to: data/outputs/{Path(image_path).stem}_vlm_explanation.txt")
    
    if out.get('action_plan'):
        print(f"\nAction Plan:")
        print(out.get('action_plan'))
    
    return 0


if __name__ == "__main__":
    exit(main())
