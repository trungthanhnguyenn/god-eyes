from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from ..config.base_config import BaseModelConfig, Provider, ModelType

class BaseModel(ABC):
    def __init__(self, config: BaseModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.provider = config.provider
        self.model_type = config.model_type
        self.api_key = config.api_key
        self.base_url = getattr(config, 'base_url', None)

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from prompt"""
        pass

    def validate_config(self) -> bool:
        """Validate model configuration"""
        try:
            return self.config.validate()
        except ValueError:
            return False

    @property
    def is_available(self) -> bool:
        """Check if model is available for use"""
        return self.validate_config()

class TextModel(BaseModel):
    def __init__(self, config: BaseModelConfig):
        if config.model_type != ModelType.TEXT:
            raise ValueError("Config must be for TEXT model type")
        super().__init__(config)

    @abstractmethod
    async def generate_text(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        pass

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs):
        """Stream text generation"""
        pass

class VisionModel(BaseModel):
    def __init__(self, config: BaseModelConfig):
        if config.model_type != ModelType.VISION:
            raise ValueError("Config must be for VISION model type")
        super().__init__(config)

    @abstractmethod
    async def analyze_image(self, image_path: str, prompt: str = "", **kwargs) -> str:
        pass

    @abstractmethod
    async def generate_from_image(self, image_data: bytes, prompt: str, **kwargs) -> str:
        pass

class MultiModalModel(BaseModel):
    def __init__(self, config: BaseModelConfig):
        if config.model_type != ModelType.MULTIMODAL:
            raise ValueError("Config must be for MULTIMODAL model type")
        super().__init__(config)

    @abstractmethod
    async def generate_multimodal(
        self, 
        text: str, 
        images: Optional[List[Union[str, bytes]]] = None,
        **kwargs
    ) -> str:
        pass

class EmbeddingModel(BaseModel):
    def __init__(self, config: BaseModelConfig):
        if config.model_type != ModelType.EMBEDDING:
            raise ValueError("Config must be for EMBEDDING model type")
        super().__init__(config)

    @abstractmethod
    async def embed_text(self, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        pass

    @abstractmethod
    async def embed_image(self, image_data: bytes, **kwargs) -> List[float]:
        pass