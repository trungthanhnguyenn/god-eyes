from typing import Type, Dict, Any, Optional
from .model import BaseModel
from ..config.base_config import BaseModelConfig, Provider, ModelType
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class to create model instances"""
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, provider: Provider, model_type: ModelType, model_class: Type[BaseModel]):
        """Register a model class for a specific provider and type"""
        key = f"{provider.value}_{model_type.value}"
        cls._registry[key] = model_class
        logger.info(f"Registered {model_class.__name__} for {provider.value} {model_type.value}")
    
    @classmethod
    def create_model(cls, config: BaseModelConfig) -> BaseModel:
        """Create model instance based on config"""
        key = f"{config.provider.value}_{config.model_type.value}"
        
        if key not in cls._registry:
            available_keys = list(cls._registry.keys())
            raise ValueError(
                f"No model registered for {config.provider.value} {config.model_type.value}. "
                f"Available combinations: {available_keys}"
            )
        
        model_class = cls._registry[key]
        try:
            return model_class(config)
        except Exception as e:
            logger.error(f"Failed to create {model_class.__name__}: {e}")
            raise
    
    @classmethod
    def create_model_from_dict(cls, config_dict: Dict[str, Any]) -> BaseModel:
        """Create model from dictionary config"""
        # Convert string values back to enums
        if isinstance(config_dict.get('provider'), str):
            config_dict['provider'] = Provider(config_dict['provider'])
        if isinstance(config_dict.get('model_type'), str):
            config_dict['model_type'] = ModelType(config_dict['model_type'])
            
        config = BaseModelConfig(**config_dict)
        return cls.create_model(config)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """List all registered models"""
        return cls._registry.copy()
    
    @classmethod
    def is_registered(cls, provider: Provider, model_type: ModelType) -> bool:
        """Check if a combination is registered"""
        key = f"{provider.value}_{model_type.value}"
        return key in cls._registry