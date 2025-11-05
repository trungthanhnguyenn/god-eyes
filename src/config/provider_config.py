from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from dataclasses import dataclass
from .base_config import BaseModelConfig, Provider, ModelType
 
@dataclass
class OpenRouterConfig(BaseModelConfig):
    """OpenRouter specific configuration"""
 
    def __post_init__(self):
        if not self.base_url:
            self.base_url = "https://openrouter.ai/api/v1"
        if not hasattr(self, 'provider') or self.provider is None:
            self.provider = Provider.OPEN_ROUTER

class OpenAIConfig(BaseModelConfig):
    """OpenAI specific configuration"""
    organization: Optional[str] = None
    
    def __post_init__(self):
        if not self.base_url:
            self.base_url = "https://api.openai.com/v1"
        if not hasattr(self, 'provider') or self.provider is None:
            self.provider = Provider.OPENAI

@dataclass  
class AnthropicConfig(BaseModelConfig):
    """Anthropic specific configuration"""
    
    def __post_init__(self):
        if not self.base_url:
            self.base_url = "https://api.anthropic.com"
        if not hasattr(self, 'provider') or self.provider is None:
            self.provider = Provider.ANTHROPIC
    """Default Claude model"""
    
    def validate(self) -> bool:
        super().validate()
        # Anthropic-specific validation
        if not self.api_key or not self.api_key.startswith("sk-ant-"):
            raise ValueError("Anthropic API key must start with 'sk-ant-'")
        return True


@dataclass
class GoogleGeminiConfig(BaseModelConfig):
    """Google Gemini specific configuration"""
    
    def __post_init__(self):
        if not self.base_url:
            self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        if not hasattr(self, 'provider') or self.provider is None:
            self.provider = Provider.GOOGLE

@dataclass
class LocalModelConfig(BaseModelConfig):
    """Local model configuration"""
    model_path: str = ""
    gpu_id: Optional[int] = None
    num_threads: int = 8
    quantization: Optional[str] = None
    
    def __post_init__(self):
        if not self.base_url:
            self.base_url = "http://localhost:11434"  # Default Ollama port
        if not hasattr(self, 'provider') or self.provider is None:
            self.provider = Provider.LOCAL

@dataclass
class GoogleImageSearchConfig(BaseModelConfig):
    """
    Configuration for Google Image Search
    """
    provider: Provider = Provider.GOOGLE
    model_type: ModelType = ModelType.MULTIMODAL
    search_engine_id: str = ""
    num_results: int = 10