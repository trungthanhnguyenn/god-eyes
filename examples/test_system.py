import asyncio
import os
from dotenv import load_dotenv
from src.base.factory import ModelFactory
from src.config.base_config import BaseModelConfig, Provider, ModelType

# Load environment variables from .env file
load_dotenv()

# Import registry to auto-register all providers
from src.providers import registry

def test_registration():
    """Test provider registration"""
    print("Testing provider registration...")
    print("Available models:")
    models = ModelFactory.list_available_models()
    for key, model_class in models.items():
        print(f"  ✓ {key}: {model_class.__name__}")
    
    print(f"\nTotal registered models: {len(models)}")
    return len(models) > 0

def test_config_validation():
    """Test config validation without API calls"""
    print("\nTesting config validation...")
    
    # Test valid config
    try:
        config = BaseModelConfig(
            model_name="anthropic/claude-3-haiku",
            provider=Provider.OPEN_ROUTER,
            model_type=ModelType.TEXT,
            api_key="test-key",
            temperature=0.5,
            max_tokens=100
        )
        
        is_valid = config.validate()
        print(f"Valid config validation: {is_valid}")
        
    except Exception as e:
        print(f"Valid config failed: {e}")
        return False
    
    # Test invalid config (missing API key for provider that needs it)
    try:
        invalid_config = BaseModelConfig(
            model_name="gpt-4",
            provider=Provider.OPENAI,
            model_type=ModelType.TEXT,
            api_key=None  # Missing API key
        )
        
        invalid_config.validate()
        print(f"Invalid config should have failed!")
        return False
        
    except ValueError as e:
        print(f"Invalid config correctly rejected: {e}")

    return True

def test_model_creation():
    """Test model creation without API calls"""
    print("\nTesting model creation...")

    try:
        config = BaseModelConfig(
            model_name="anthropic/claude-3-haiku",
            provider=Provider.OPEN_ROUTER,
            model_type=ModelType.TEXT,
            api_key="test-key-123",
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7
        )
        
        # This should work without API key validation
        model = ModelFactory.create_model(config)
        print(f"Model created: {type(model).__name__}")
        print(f"Model name: {model.model_name}")
        print(f"Provider: {model.provider}")
        print(f"Model type: {model.model_type}")

        return True
        
    except Exception as e:
        print(f"Model creation failed: {e}")
        return False

async def test_with_real_api():
    """Test with real API if key is available"""
    print("\nTesting with real API (if available)...")

    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f" Debug - API key loaded: {'Yes' if api_key else 'No'}")
    if api_key:
        print(f"Debug - API key prefix: {api_key[:20]}...")
    
    if not api_key:
        print("No OPENROUTER_API_KEY found - skipping real API test")
        print("Make sure .env file is in project root with OPENROUTER_API_KEY")
        return True
    
    try:
        config = BaseModelConfig(
            model_name="anthropic/claude-3-haiku",
            provider=Provider.OPEN_ROUTER,
            model_type=ModelType.TEXT,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_tokens=50
        )
        
        model = ModelFactory.create_model(config)
        print(f"Model available: {model.is_available}")
        
        if model.is_available:
            response = await model.generate("Say 'Hello' in one word")
            print(f"API Response: {response}")
            return True
        else:
            print("Model not available (config validation failed)")
            return False
            
    except Exception as e:
        print(f"Real API test failed: {e}")
        return False

async def main():
    print("God Eyes - Model System Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Registration
    if test_registration():
        success_count += 1
    
    # Test 2: Config validation
    if test_config_validation():
        success_count += 1
    
    # Test 3: Model creation
    if test_model_creation():
        success_count += 1
    
    # Test 4: Real API (optional)
    if await test_with_real_api():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {success_count}/{total_tests} passed")

    if success_count == total_tests:
        print("All tests passed! System is working correctly.")
    elif success_count >= 3:
        print("Core system working. API test may need environment setup.")
    else:
        print("Some core tests failed. Check your implementation.")

if __name__ == "__main__":
    asyncio.run(main())
