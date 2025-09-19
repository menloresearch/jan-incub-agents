import openai
from typing import List, Dict, Any, Optional
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class OpenAIApiWrapper:
    """Wrapper for OpenAI API that can point to different endpoints (OpenAI, VLLM, OpenRouter)."""
    
    def __init__(
        self,
        model_id: str = None,
        base_url: str = None,
        api_key: Optional[str] = None,
        sampling_params: Dict[str, Any] = None
    ):
        self.model_id = model_id or os.getenv("MODEL_ID", "gpt-4")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.sampling_params = sampling_params or {}
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url
        )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate text using the LLM."""
        
        # Merge sampling parameters
        params = {**self.sampling_params, **kwargs}
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **params
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def generate_with_system(
        self,
        system_prompt: str,
        user_messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Generate text with system prompt and user messages."""
        
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(user_messages)
        
        return self.generate(messages, **kwargs)