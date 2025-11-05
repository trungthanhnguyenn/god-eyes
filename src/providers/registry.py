"""Import all providers to register them with the factory"""

try:
    from . import openrouter
    print("OpenRouter provider loaded")
except ImportError as e:
    print(f"Failed to load OpenRouter: {e}")

# Add more providers
try:
    from . import openai
    print("OpenAI provider loaded") 
except ImportError as e:
    print(f"Failed to load OpenAI: {e}")

try:
    from . import anthropic
    print("Anthropic provider loaded")
except ImportError as e:
    print(f"Failed to load Anthropic: {e}")