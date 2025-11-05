from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from .base_config import BaseModelConfig, Provider, ModelType


@dataclass
class TextModelConfig(BaseModelConfig):
    """
    Config for Text-only models (e.g., GPT-4, Claude)
    Inherits from BaseModelConfig and adds specific fields
    """
    
    model_type: ModelType = ModelType.TEXT
    """Default type is TEXT"""

    system_prompt: Optional[str] = None
    """Default system prompt for this model"""

    streaming_enabled: bool = True
    """Is streaming enabled?"""

    top_p: float = 0.9
    """Nucleus sampling parameter"""


@dataclass
class VisionModelConfig(BaseModelConfig):
    """
    Config for Vision models (image analysis)
    """
    
    model_type: ModelType = ModelType.VISION
    """Default type is VISION"""

    supported_formats: List[str] = field(default_factory=lambda: ["JPEG", "PNG", "GIF", "WebP"])
    """Supported image formats"""

    max_image_size: int = 20  # MB
    """Maximum image size (MB)"""
    
    image_detail: str = "auto"  # "low", "high", "auto"
    """Image processing detail"""

    ocr_enabled: bool = False
    """Is OCR enabled?"""


@dataclass
class MultiModalModelConfig(BaseModelConfig):
    """
    Config for Multimodal models (text + image)
    """
    
    model_type: ModelType = ModelType.MULTIMODAL
    """Default type is MULTIMODAL"""
    
    vision_enabled: bool = True
    """Is image processing supported?"""

    supported_formats: List[str] = field(default_factory=lambda: ["JPEG", "PNG", "GIF", "WebP"])
    """Supported image formats"""

    max_image_size: int = 20  # MB
    """Maximum image size"""
    
    system_prompt: Optional[str] = None
    """System prompt"""


@dataclass
class EmbeddingModelConfig(BaseModelConfig):
    """
    Config for Embedding models
    """
    
    model_type: ModelType = ModelType.EMBEDDING
    """Default type is EMBEDDING"""

    embedding_dimension: int = 1536
    """Dimensionality of embedding vectors"""

    batch_size: int = 100
    """Batch size for embedding multiple texts"""

    normalize_embeddings: bool = True
    """Normalize embeddings vector?"""

@dataclass
class ImageSearchConfig(BaseModelConfig):
    """
    Config for Image Search models
    """
    
    model_type: ModelType = ModelType.MULTIMODAL
    """Default type is MULTIMODAL"""
    
    search_engine_id: str = ""
    """Search engine ID for Google Custom Search"""
    
    num_results: int = 10
    """Number of results to return"""
