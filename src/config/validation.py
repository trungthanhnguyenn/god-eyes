from typing import Union
import os
from pathlib import Path
import yaml
from .base_config import Provider, ModelType
from .provider_config import (
    OpenAIConfig,
    AnthropicConfig,
    GoogleGeminiConfig,
    LocalModelConfig
)


class ConfigValidator:
    """
    Helper class để validate và load configurations
    """
    
    @staticmethod
    def load_from_env() -> dict:
        """
        Load config từ environment variables
        Ví dụ:
            MODEL_PROVIDER=openai
            MODEL_NAME=gpt-4
            OPENAI_API_KEY=sk-...
        """
        return {
            "provider": os.getenv("MODEL_PROVIDER"),
            "model_name": os.getenv("MODEL_NAME"),
            "api_key": os.getenv("API_KEY"),
            "temperature": float(os.getenv("TEMPERATURE", 0.7)),
            "max_tokens": int(os.getenv("MAX_TOKENS", 1024)),
        }
    
    @staticmethod
    def load_from_yaml(config_path: str) -> dict:
        """
        Load config từ YAML file
        Ví dụ config.yaml:
            models:
              text_model:
                provider: openai
                model_name: gpt-4
                api_key: ${OPENAI_API_KEY}  # Từ env var
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def create_config(
        provider: Union[str, Provider],
        model_name: str,
        api_key: str = None,
        **kwargs
    ) -> Union[OpenAIConfig, AnthropicConfig, GoogleGeminiConfig, LocalModelConfig]:
        """
        Factory function để tạo config dựa trên provider
        """
        
        # Convert string to Provider enum nếu cần
        if isinstance(provider, str):
            provider = Provider[provider.upper()]
        
        # Tạo config object phù hợp
        if provider == Provider.OPENAI:
            config = OpenAIConfig(
                model_name=model_name,
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                **kwargs
            )
        
        elif provider == Provider.ANTHROPIC:
            config = AnthropicConfig(
                model_name=model_name,
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
                **kwargs
            )
        
        elif provider == Provider.GOOGLE:
            config = GoogleGeminiConfig(
                model_name=model_name,
                api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                **kwargs
            )
        
        elif provider == Provider.LOCAL:
            config = LocalModelConfig(
                model_name=model_name,
                api_key=None,  # Local models không cần API key
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Validate config
        config.validate()
        
        return config
