from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class Provider(Enum):
    """Model providers"""
    OPEN_ROUTER = "open_router"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    LOCAL = "local"


class ModelType(Enum):
    """Model types"""
    TEXT = "text"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"


@dataclass
class BaseModelConfig:
    """
    Base configuration for models
    """
    
    model_name: str
    """Name of the model (e.g., 'gpt-4', 'claude-3-sonnet', 'llama2')"""

    provider: Provider
    """Provider of the model"""

    model_type: ModelType
    """Type of model"""

    api_key: Optional[str] = None
    """API key (if required by provider)"""

    base_url: Optional[str] = None
    """Base URL for API (if different from default)"""

    timeout: int = 30
    """Timeout for requests (seconds)"""

    max_retries: int = 3
    """Number of retries if request fails"""

    temperature: float = 0.7
    """Creativity level (0.0 - 1.0)"""
    
    max_tokens: int = 1024
    """Max tokens in response"""

    additional_params: Dict[str, Any] = field(default_factory=dict)
    """Other additional params"""

    def validate(self) -> bool:
        """
        Validate configuration
        Return True if valid, else raise ValueError
        """
        # Check required fields
        if not self.model_name:
            raise ValueError("model_name cannot be empty")

        if not self.provider:
            raise ValueError("provider cannot be empty")

        # Providers that require API key
        if self.provider in [Provider.OPENAI, Provider.ANTHROPIC, Provider.GOOGLE]:
            if not self.api_key:
                raise ValueError(f"{self.provider.value} requires API key")

        # Validate ranges
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")

        return True
