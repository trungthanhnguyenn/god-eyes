import asyncio
import os
from dotenv import load_dotenv
from src.base.factory import ModelFactory
from src.base.model import MultiModalModel
from src.config.base_config import BaseModelConfig, Provider, ModelType
from src.config.provider_config import OpenRouterConfig

# Load environment variables from .env file
load_dotenv()

# Import registry to auto-register all providers
from src.providers import registry

async def test_openrouter_with_config():
    """Test OpenRouter integration using config"""
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        print(f"API Key found: {'Yes' if api_key else 'No'}")
        
        if not api_key:
            print("Please set OPENROUTER_API_KEY environment variable")
            return
            
        # Method 1: Using BaseModelConfig
        config = BaseModelConfig(
            model_name="anthropic/claude-3-haiku",
            provider=Provider.OPEN_ROUTER,
            model_type=ModelType.TEXT,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_tokens=500,
            temperature=0.5
        )
        
        print(f"Config validation: {config.validate()}")
        
        model = ModelFactory.create_model(config)
        print(f"Model available: {model.is_available}")
        
        if model.is_available:
            response = await model.generate("Say hello in Vietnamese")
            print(f"Response: {response}")
        else:
            print("Model not available - check configuration")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def test_openrouter_specific_config():
    """Test using OpenRouter specific config"""
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Skipping multimodal test - no API key found")
            return
            
        # Method 2: Using OpenRouterConfig (auto-sets base_url)
        # Use the VLM model from .env file
        vlm_model = os.getenv("VLM_MODEL_NAME", "")
        config = OpenRouterConfig(
            model_name=vlm_model,
            provider=Provider.OPEN_ROUTER,
            model_type=ModelType.MULTIMODAL,
            api_key=api_key,
            temperature=0.7
        )   
        
        model = ModelFactory.create_model(config)
        
        if model.is_available:
            # Method 1: Text + Image URL
            if hasattr(model, 'generate_multimodal'):
                # Example with image URL
                image_urls = [
                    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Sunset_at_the_Golden_Gate_Bridge.jpg/640px-Sunset_at_the_Golden_Gate_Bridge.jpg"
                ]
                
                response = await getattr(model, 'generate_multimodal')(
                    "What do you see in this image? Describe it in detail.",
                    images=image_urls
                )
                print(f"Vision response: {response}")
                
            else:
                # Fallback to text-only
                response = await model.generate("What would you see in a beautiful sunset?")
                print(f"Text response: {response}")
        else:
            print("Vision model not available")
            
    except Exception as e:
        print(f"Vision error: {e}")
        import traceback
        traceback.print_exc()

async def test_config_from_dict():
    """Test creating model from dictionary config"""
    try:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Skipping dict test - no API key")
            return
            
        config_dict = {
            "model_name": "anthropic/claude-3-haiku",
            "provider": "open_router",  # String value
            "model_type": "text",       # String value
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": 0.3,
            "max_tokens": 200
        }
        
        model = ModelFactory.create_model_from_dict(config_dict)
        response = await model.generate("Explain AI in one sentence")
        print(f"Dict config response: {response}")
        
    except Exception as e:
        print(f"Dict config error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    print("Available models:")
    for key, model_class in ModelFactory.list_available_models().items():
        print(f"  {key}: {model_class.__name__}")
    
    print("\n1. Testing OpenRouter with BaseModelConfig...")
    await test_openrouter_with_config()
    
    print("\n2. Testing OpenRouter with specific config...")
    await test_openrouter_specific_config()
    
    print("\n3. Testing config from dictionary...")
    await test_config_from_dict()

if __name__ == "__main__":
    asyncio.run(main())
