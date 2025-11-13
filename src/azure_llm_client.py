"""
Azure OpenAI LLM Client
Provides text generation using Azure OpenAI (o1-mini, GPT-4, etc.)
"""

import logging
from typing import List, Dict, Optional
from openai import AzureOpenAI

logger = logging.getLogger(__name__)


class AzureLLMClient:
    """Client for Azure OpenAI models (o1-mini, GPT-4, etc.)"""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2024-02-15-preview",
        deployment_name: str = "o1-mini",
        max_tokens: int = 512,
        temperature: float = 0.7
    ):
        """
        Initialize Azure OpenAI client
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version
            deployment_name: Deployment name (e.g., o1-mini, gpt-4)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        try:
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version
            )
            logger.info(f"✓ Azure OpenAI client initialized: {deployment_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Generate text response from messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Override default max tokens
            temperature: Override default temperature
            
        Returns:
            Generated text response
        """
        tokens = max_tokens or self.max_tokens
        temp = temperature if temperature is not None else self.temperature
        
        logger.debug(f"Azure OpenAI request: {self.deployment_name}, max_completion_tokens={tokens}")
        
        try:
            # Build request parameters
            # Use max_completion_tokens for newer models (gpt-5-nano, o1, etc.)
            kwargs = {
                "model": self.deployment_name,
                "messages": messages,
                "max_completion_tokens": tokens
            }
            
            # Only include temperature for models that support it
            # Reasoning models (o1, gpt-5-nano) don't support custom temperature
            # Omit temperature parameter to use model defaults
            
            response = self.client.chat.completions.create(**kwargs)
            
            generated_text = response.choices[0].message.content
            return generated_text.strip()
        
        except Exception as e:
            logger.error(f"Azure OpenAI generation failed: {str(e)}")
            raise
    
    def generate_action_plan(
        self,
        analysis_text: str,
        user_command: str,
        system_prompt: str = None
    ) -> str:
        """
        Generate action plan from OmniParser analysis
        
        Args:
            analysis_text: Parsed UI element analysis
            user_command: User's desired action
            system_prompt: Optional system prompt override
            
        Returns:
            Generated action plan
        """
        # Check if this is a reasoning model (o1, gpt-5-nano) that doesn't support system messages
        is_reasoning_model = any(x in self.deployment_name.lower() for x in ['o1', 'gpt-5', 'reasoning'])
        
        if system_prompt is None:
            system_prompt = """You are a UI automation assistant that generates executable action commands.
Analyze the UI elements and user task to create a precise action sequence.

Use these command formats:
- Click(x, y, element_id, "element_description")
- Type(x, y, element_id, "field_name", "text_to_enter")
- Submit(x, y, element_id, "button_name")"""
        
        if is_reasoning_model:
            # Reasoning models (o1, gpt-5-nano) don't support system messages
            # Combine everything into a single user message
            combined_prompt = f"""{system_prompt}

Based on the following UI analysis and user command, generate an action plan.

UI Analysis:
{analysis_text}

User Command: {user_command}

Generate a numbered list of actions to accomplish this task."""
            
            messages = [
                {"role": "user", "content": combined_prompt}
            ]
        else:
            # Regular models (GPT-4, GPT-3.5) support system messages
            user_prompt = f"""Based on the following UI analysis and user command, generate an action plan.

UI Analysis:
{analysis_text}

User Command: {user_command}

Generate a numbered list of actions to accomplish this task."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        
        action_plan = self.generate(messages)
        return action_plan
    
    def run_with_omniparser(
        self,
        analysis_path: str,
        user_command: str,
        system_prompt: str = None
    ) -> Dict[str, str]:
        """
        Compatible interface for OmniParser integration
        
        Args:
            analysis_path: Path to OmniParser analysis file
            user_command: User's command
            system_prompt: Optional system prompt
            
        Returns:
            Dict with action_plan and metadata
        """
        logger.info(f"Generating action plan for: {user_command[:50]}...")
        
        # Read analysis file
        with open(analysis_path, 'r') as f:
            analysis_text = f.read()
        
        # Generate action plan
        action_plan = self.generate_action_plan(analysis_text, user_command, system_prompt)
        
        return {
            "action_plan": action_plan,
            "user_command": user_command,
            "analysis_path": analysis_path,
            "model": f"azure/{self.deployment_name}"
        }
