"""
Vision Model Example - Demonstrating how to use VLM models with different image inputs
"""
import asyncio
import os
from dotenv import load_dotenv
from src.base.factory import ModelFactory
from src.config.base_config import BaseModelConfig, Provider, ModelType
from src.config.provider_config import OpenRouterConfig

# Load environment variables
load_dotenv()

# Import registry to auto-register all providers
from src.providers import registry

async def test_vision_with_url():
    """Test vision model with image URLs"""
    print("Testing Vision Model with Image URLs...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    vlm_model = os.getenv("VLM_MODEL_NAME", "")
    
    if not api_key:
        print("Please set OPENROUTER_API_KEY in .env file")
        return
    
    try:
        config = BaseModelConfig(
            model_name=vlm_model,
            provider=Provider.OPEN_ROUTER,
            model_type=ModelType.MULTIMODAL,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=500
        )
        
        model = ModelFactory.create_model(config)
        
        if model.is_available and hasattr(model, 'generate_multimodal'):
            # Test 1: Single image
            print("\nSingle Image Analysis:")
            response = await getattr(model, 'generate_multimodal')(
                "Describe this image in detail. What do you see?",
                images=["https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Sunset_at_the_Golden_Gate_Bridge.jpg/640px-Sunset_at_the_Golden_Gate_Bridge.jpg"]
            )
            print(f"Response: {response}")
            
            # Test 2: Different image with specific questions
            print("\nSpecific Question About Image:")
            response2 = await getattr(model, 'generate_multimodal')(
                "What architectural style is this building? What materials can you identify?",
                images=["https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Eiffel_Tower_from_north_Avenue_de_New_York%2C_Aug_2010.jpg/640px-Eiffel_Tower_from_north_Avenue_de_New_York%2C_Aug_2010.jpg"]
            )
            print(f"Response: {response2}")
            
            # Test 3: Multiple images comparison (if supported)
            print("\nMultiple Images:")
            try:
                response3 = await getattr(model, 'generate_multimodal')(
                    "Compare these two famous landmarks. What are the similarities and differences?",
                    images=[
                        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Sunset_at_the_Golden_Gate_Bridge.jpg/320px-Sunset_at_the_Golden_Gate_Bridge.jpg",
                        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/Eiffel_Tower_from_north_Avenue_de_New_York%2C_Aug_2010.jpg/320px-Eiffel_Tower_from_north_Avenue_de_New_York%2C_Aug_2010.jpg"
                    ]
                )
                print(f"Response: {response3}")
            except Exception as e:
                print(f"Multiple images not supported: {e}")
        
        else:
            print("Vision model not available or doesn't support multimodal")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

async def test_vision_with_base64():
    """Test vision model with base64 encoded images"""
    print("\nTesting Vision Model with Base64 Images...")

    # Note: This is for demonstration - you would need to implement base64 encoding
    # in your OpenRouter provider if it supports it
    print("Base64 image support would need to be implemented in the provider")

async def test_different_prompts():
    """Test various types of vision prompts"""
    print("\nTesting Different Prompt Types...")

    api_key = os.getenv("OPENROUTER_API_KEY")
    vlm_model = os.getenv("VLM_MODEL_NAME", "nvidia/nemotron-nano-12b-v2-vl:free")
    
    if not api_key:
        return
    
    config = BaseModelConfig(
        model_name=vlm_model,
        provider=Provider.OPEN_ROUTER,
        model_type=ModelType.MULTIMODAL,
        api_key=api_key,
        temperature=0.7
    )
    
    model = ModelFactory.create_model(config)
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/HTML5_logo_and_wordmark.svg/640px-HTML5_logo_and_wordmark.svg.png"
    
    if model.is_available and hasattr(model, 'generate_multimodal'):
        prompts = [
            "What do you see in this image?",
            "Describe the colors and shapes.",
            "What text or symbols are visible?", 
            "What is this logo used for?",
            "Explain this image as if describing it to someone who can't see it."
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}️⃣ Prompt: {prompt}")
            try:
                response = await getattr(model, 'generate_multimodal')(prompt, images=[image_url])
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")

async def main():
    print("Vision Model Testing Suite")
    print("=" * 50)
    
    await test_vision_with_url()
    await test_vision_with_base64()
    await test_different_prompts()
    
    print("\nVision testing complete!")

if __name__ == "__main__":
    asyncio.run(main())
