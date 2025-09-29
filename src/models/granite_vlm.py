"""Granite VLM integration for UI understanding."""

import torch
from PIL import Image
from vllm import LLM, SamplingParams
from typing import Dict, Any
from ..config import GRANITE_MODEL_PATH, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEVICE


class GraniteVLM:
    def __init__(self):
        self.model_path = GRANITE_MODEL_PATH
        self.device = DEVICE
        self.model = None
        self.sampling_params = None
        self._initialize_model()
    
    def _initialize_model(self):
        self.model = LLM(model=self.model_path)
        
        self.sampling_params = SamplingParams(
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    
    def analyze_ui(self, image: Image.Image, question: str = None) -> str:
        if question is None:
            question = "Describe the UI elements visible in this image and their likely functions."
        
        image_token = "<image>"
        system_prompt = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        
        prompt = f"{system_prompt}<|user|>\n{image_token}\n{question}\n<|assistant|>\n"
        
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image.convert("RGB"),
            }
        }
        
        outputs = self.model.generate(inputs, self.sampling_params)
        
        return outputs[0].outputs[0].text.strip()
    
    def identify_element(self, image: Image.Image, element_description: str) -> str:
        question = f"Where is the {element_description} located in this UI? Describe its position and characteristics."
        return self.analyze_ui(image, question)
    
    def suggest_action(self, image: Image.Image, goal: str) -> str:
        question = f"To achieve the goal: '{goal}', what UI element should I interact with and how?"
        return self.analyze_ui(image, question)
