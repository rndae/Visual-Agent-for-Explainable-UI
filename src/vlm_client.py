import os
from typing import List, Optional, Dict, Any

from PIL import Image


class QwenVLMClient:
    """Lightweight client wrapper for Qwen3-VL via Hugging Face Transformers.

    The class loads model and processor lazily and exposes concise helpers used
    by the omniparser workflow (summarize image, consume omniparser analysis,
    and produce action mappings / explanations).
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct", device_map: str = "auto", dtype: str = "auto", use_flash_attention: bool = False, auto_load: bool = True):
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.use_flash_attention = use_flash_attention
        self.model = None
        self.processor = None
        if auto_load:
            self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        except Exception as exc:
            raise RuntimeError("Missing dependencies for VLM client: install 'transformers' and 'torch' before using this module") from exc

        kwargs = {"dtype": "auto", "device_map": self.device_map}
        if self.use_flash_attention:
            kwargs.update({"dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"})

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def _generate(self, messages: List[Dict[str, Any]], max_new_tokens: int = 128) -> str:
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text[0] if output_text else ""

    def summarize_image(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Provide a concise one-sentence summary of this UI screenshot, mentioning sections if relevant."}]}
        ]
        return self._generate(messages)

    def run_with_omniparser(self, image_path: str, analysis_txt_path: Optional[str] = None, user_command: Optional[str] = None, save_to: Optional[str] = None) -> Dict[str, Any]:
        base = os.path.splitext(os.path.basename(image_path))[0]
        if analysis_txt_path is None:
            analysis_txt_path = os.path.join("data", "outputs", f"{base}_analysis.txt")
        if save_to is None:
            save_to = os.path.join("data", "outputs", f"{base}_vlm_explanation.txt")

        summary = self.summarize_image(image_path)

        analysis_text = ""
        if os.path.exists(analysis_txt_path):
            with open(analysis_txt_path, "r", encoding="utf-8") as fh:
                analysis_text = fh.read()

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a UI automation assistant that generates executable action commands based on visual analysis and user tasks."}]},
            {"role": "user", "content": [{"type": "image", "image": Image.open(image_path).convert("RGB")}, {"type": "text", "text": f"UI Summary: {summary}"}]},
        ]

        if analysis_text:
            messages.append({"role": "user", "content": [{"type": "text", "text": "Detected UI Elements:\n" + analysis_text}]})

        suggestions = ""
        action_plan = ""
        
        if user_command:
            # Generate structured action commands
            action_prompt = f"""User Task: {user_command}

Generate executable UI automation commands in this format:
Click(x, y, element_id, "element_description")
Type(x, y, element_id, "field_name", "text_to_enter")
Submit(x, y, element_id, "button_name")

Rules:
1. Use exact coordinates from the element analysis
2. Reference Element IDs from the analysis
3. One command per line
4. Commands must be executable in sequence

Generate the action sequence now:"""
            
            messages.append({"role": "user", "content": [{"type": "text", "text": action_prompt}]})
            action_plan = self._generate(messages, max_new_tokens=256)
        else:
            # Generate general suggestions
            prompt_next = "Based on the UI elements, list possible actions (one per line, max 20 words each):"
            messages.append({"role": "user", "content": [{"type": "text", "text": prompt_next}]})
            suggestions = self._generate(messages)

        out = {"summary": summary, "suggestions": suggestions, "action_plan": action_plan}

        os.makedirs(os.path.dirname(save_to), exist_ok=True)
        with open(save_to, "w", encoding="utf-8") as fh:
            fh.write("SUMMARY:\n")
            fh.write(summary + "\n\n")
            
            if action_plan:
                fh.write("ACTION PLAN:\n")
                fh.write(action_plan + "\n\n")
            
            if suggestions:
                fh.write("SUGGESTIONS:\n")
                fh.write(suggestions + "\n\n")

        return out


def demo_run_default():
    image = os.path.join("images", "form-example-2.png")
    client = QwenVLMClient(auto_load=True)
    base = os.path.splitext(os.path.basename(image))[0]
    analysis = os.path.join("data", "outputs", f"{base}_analysis.txt")
    user_instructions = "Fill form with: Jose de la Rosa, 242343111, josedelarosaroja@jose.com, 2444 pine st, Seal Beach, CA, 90740"
    return client.run_with_omniparser(image, analysis_txt_path=analysis, user_instructions=user_instructions)
