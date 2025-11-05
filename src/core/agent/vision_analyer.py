import os
from dotenv import load_dotenv

from src.base.factory import ModelFactory
from src.config.base_config import Provider, ModelType
from src.config.provider_config import OpenRouterConfig

# Load environment variables
load_dotenv()

# Import registry to auto-register all providers
from src.providers import registry

class VisionAnalyzer:
    """Class to analyze images using a VLM model from OpenRouter."""

    def __init__(self):
        # Initialize the VLM model from OpenRouter
        config = OpenRouterConfig(
            model_name=os.getenv("VLM_MODEL_NAME", ""),  # Example model name
            api_key=os.getenv("OPENROUTER_API_KEY"),
            provider=Provider.OPEN_ROUTER,
            model_type=ModelType.MULTIMODAL,
            max_tokens=1000,
            temperature=0.7
        )
        self.vlm_model = ModelFactory.create_model(config)

    
    async def analyze_image(self, query: str, image_url: list[str]):
        """Analyze image with given query using VLM model."""
        if not self.vlm_model.is_available:
            raise RuntimeError("VLM model is not available")

        if not hasattr(self.vlm_model, 'generate_multimodal'):
            raise RuntimeError("Model doesn't support multimodal generation")

        response = await getattr(self.vlm_model, 'generate_multimodal')(
            query,
            images=image_url
        )
        return response