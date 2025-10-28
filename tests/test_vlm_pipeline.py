import os
from pathlib import Path

from PIL import Image
import importlib.util


spec = importlib.util.spec_from_file_location("vlm_client", os.path.join("src", "vlm_client.py"))
vlm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlm)
QwenVLMClient = vlm.QwenVLMClient


def test_run_with_mocked_generation(tmp_path):
    img_dir = Path("images")
    img_dir.mkdir(exist_ok=True)
    img_path = img_dir / "form-example-2.png"

    if not img_path.exists():
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img.save(img_path)

    base = img_path.stem
    outputs_dir = Path("data") / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    analysis_file = outputs_dir / f"{base}_analysis.txt"
    analysis_content = "UI ELEMENT ANALYSIS REPORT\nTotal Elements Detected: 0\n"
    analysis_file.write_text(analysis_content, encoding="utf-8")

    client = QwenVLMClient(auto_load=False)

    client._generate = lambda messages, max_new_tokens=128: "MOCKED_GENERATION_OUTPUT"

    res = client.run_with_omniparser(str(img_path), analysis_txt_path=str(analysis_file), user_instructions="Fill with test data")

    out_file = outputs_dir / f"{base}_vlm_explanation.txt"
    assert out_file.exists(), "VLM explanation file should be created"
    text = out_file.read_text(encoding="utf-8")
    assert "SUMMARY:" in text
    assert "SUGGESTIONS:" in text
