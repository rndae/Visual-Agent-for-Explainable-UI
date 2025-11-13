"""
LLM Client for Text-Based UI Automation
Supports Gemma 2 9B (default), Llama 3.1 8B, and other text generation models
Provides fast text-only inference without vision requirements
"""

import logging
from typing import Optional, List, Dict, Any
import torch
import transformers
from transformers import BitsAndBytesConfig
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Universal client for text generation models (Gemma, Llama, etc.)
    Text-only reasoning for UI automation tasks
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-2-9b",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 512,
        quantization: Optional[str] = None,
        auto_load: bool = True
    ):
        """
        Initialize LLM client
        
        Args:
            model_name: HuggingFace model identifier
                - google/gemma-2-9b (default)
                - google/gemma-2-9b-it (instruction-tuned)
                - meta-llama/Meta-Llama-3.1-8B-Instruct
            device: Device placement (auto, cuda, cpu, mps)
            torch_dtype: Torch data type (bfloat16, float16, float32)
            max_new_tokens: Maximum tokens to generate
            quantization: Quantization mode (None, "8bit", "4bit")
            auto_load: Automatically load model on init
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.quantization = quantization
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)
        
        self.pipeline: Optional[transformers.Pipeline] = None
        self.model = None
        self.tokenizer = None
        
        if auto_load:
            self._load()
    
    def _load(self) -> None:
        """Load the model and create pipeline"""
        logger.info(f"Loading LLM model: {self.model_name}")
        logger.info(f"Device: {self.device}, dtype: {self.torch_dtype}")
        
        if self.quantization:
            logger.info(f"Quantization: {self.quantization}")
        
        try:
            if self.quantization:
                # Load model with quantization using BitsAndBytesConfig
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                if self.quantization == "8bit":
                    logger.info("Using 8-bit quantization (int8)")
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                elif self.quantization == "4bit":
                    logger.info("Using 4-bit quantization")
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                else:
                    raise ValueError(f"Unsupported quantization: {self.quantization}")
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map=self.device if self.device != "auto" else "auto",
                )
                
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                # Create pipeline with pre-loaded model
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
            else:
                # Standard pipeline loading (no quantization)
                model_kwargs = {"torch_dtype": self.torch_dtype}
                
                self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=self.model_name,
                    model_kwargs=model_kwargs,
                    device_map=self.device if self.device != "auto" else "auto",
                )
            
            logger.info("✓ LLM model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load LLM model: {str(e)}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text response from messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Override default max tokens
            temperature: Sampling temperature (0.0-1.0)
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated text response
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call _load() first.")
        
        tokens = max_new_tokens or self.max_new_tokens
        
        logger.debug(f"Generating with max_tokens={tokens}, temp={temperature}")
        
        try:
            # Convert messages to plain text format for models without chat template
            # Format: System: <system>\n\nUser: <user>\n\nAssistant:
            prompt = self._format_messages(messages)
            
            outputs = self.pipeline(
                prompt,
                max_new_tokens=tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0
            )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"]
            
            # Remove the prompt from the output
            response = generated_text[len(prompt):].strip()
            return response
        
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages into plain text prompt for models without chat template
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add Assistant: prefix for the model to continue
        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        return prompt
    
    def generate_action_plan(
        self,
        analysis_text: str,
        user_command: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate UI automation action plan from analysis and command
        
        Args:
            analysis_text: OmniParser UI element analysis
            user_command: User's automation task description
            system_prompt: Optional custom system prompt
            
        Returns:
            Action plan as string
        """
        if system_prompt is None:
            system_prompt = """You are a UI automation assistant that generates executable action commands.
Analyze the UI elements and user task to create a precise action sequence.

Use these command formats:
- Click(x, y, element_id, "element_description")
- Type(x, y, element_id, "field_name", "text_to_enter")
- Submit(x, y, element_id, "button_name")

Be precise with coordinates and element IDs from the analysis."""
        
        user_message = f"""UI Elements Detected:
{analysis_text}

User Task: {user_command}

Generate the executable action sequence:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        logger.info(f"Generating action plan for: {user_command[:50]}...")
        action_plan = self.generate(messages, temperature=0.3)  # Lower temp for consistency
        logger.info("✓ Action plan generated")
        
        return action_plan
    
    def run_with_omniparser(
        self,
        analysis_txt_path: Optional[str] = None,
        analysis_text: Optional[str] = None,
        user_command: str = ""
    ) -> Dict[str, Any]:
        """
        Generate action plan from OmniParser analysis
        Compatible with VLM client interface
        
        Args:
            analysis_txt_path: Path to analysis text file
            analysis_text: Direct analysis text (alternative to file)
            user_command: User's automation task
            
        Returns:
            Dict with action_plan and metadata
        """
        # Load analysis text
        if analysis_text is None:
            if analysis_txt_path is None:
                raise ValueError("Either analysis_txt_path or analysis_text must be provided")
            
            analysis_path = Path(analysis_txt_path)
            if not analysis_path.exists():
                raise FileNotFoundError(f"Analysis file not found: {analysis_txt_path}")
            
            with open(analysis_path, 'r') as f:
                analysis_text = f.read()
        
        # Generate action plan
        action_plan = self.generate_action_plan(analysis_text, user_command)
        
        return {
            "action_plan": action_plan,
            "model": self.model_name,
            "mode": "llm"  # Distinguish from VLM mode
        }
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.pipeline is not None
    
    def unload(self) -> None:
        """Unload model to free memory"""
        if self.pipeline is not None:
            logger.info("Unloading Llama model...")
            del self.pipeline
            self.pipeline = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("✓ Model unloaded")
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded() else "not loaded"
        quant = f", {self.quantization}" if self.quantization else ""
        return f"LLMClient(model={self.model_name}{quant}, status={status})"


# Backward compatibility alias
LlamaLLMClient = LLMClient


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with Gemma (default)
    print("Testing Gemma 2 9B...")
    client = LLMClient(model_name="google/gemma-2-9b", auto_load=True)
    
    # Test generation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is UI automation?"}
    ]
    
    response = client.generate(messages, max_new_tokens=100)
    print(f"\nResponse: {response}")
