# God Eyes 👁️

**MultiModal AI System for Image Analysis and Search**

A flexible, provider-agnostic framework for working with Vision-Language Models (VLMs) and AI-powered image search. Built with a clean architecture that makes it easy to switch between different AI providers.

## ✨ Features

- **Multi-Provider Support**: OpenRouter, OpenAI, Anthropic, Google Gemini
- **Vision-Language Models**: Analyze images with AI
- **Image Search**: Google Custom Search integration
- **Clean Architecture**: Factory pattern with config-based initialization
- **Async/Await**: Built for performance with async operations
- **Jupyter Ready**: Works seamlessly in notebooks
- **Type-Safe**: Full type hints and validation

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/trungthanhnguyenn/god-eyes.git
cd god-eyes

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or use your favorite editor
```

**Required API Keys:**
- **OpenRouter**: Get from [openrouter.ai](https://openrouter.ai/)
- **Google Custom Search**: Get from [Google Cloud Console](https://console.cloud.google.com/)

### 3. Basic Usage

```python
import asyncio
import os
from dotenv import load_dotenv
from src.base.factory import ModelFactory
from src.config.base_config import BaseModelConfig, Provider, ModelType

# Load environment
load_dotenv()

# Import providers
from src.providers import registry

async def main():
    # Create VLM model
    config = BaseModelConfig(
        model_name="nvidia/nemotron-nano-12b-v2-vl:free",
        provider=Provider.OPEN_ROUTER,
        model_type=ModelType.MULTIMODAL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7
    )
    
    model = ModelFactory.create_model(config)
    
    # Analyze an image
    if hasattr(model, 'generate_multimodal'):
        response = await model.generate_multimodal(
            "What do you see in this image?",
            images=["https://example.com/image.jpg"]
        )
        print(response)

asyncio.run(main())
```

## 📚 Documentation

### Project Structure

```
god-eyes/
├── src/
│   ├── base/              # Base classes and factory
│   │   ├── model.py       # BaseModel, TextModel, VisionModel, etc.
│   │   └── factory.py     # ModelFactory for creating instances
│   ├── config/            # Configuration classes
│   │   ├── base_config.py # BaseModelConfig, Provider, ModelType enums
│   │   ├── model_config.py # Specific model configs
│   │   └── provider_config.py # Provider-specific configs
│   ├── providers/         # Provider implementations
│   │   ├── openrouter.py  # OpenRouter integration
│   │   ├── openai.py      # OpenAI integration
│   │   └── anthropic.py   # Anthropic integration
│   ├── core/
│   │   └── agent/
│   │       └── image_search.py # Google Image Search
│   └── utils/             # Utility functions
├── examples/              # Example scripts
│   ├── usage_example.py   # Basic usage
│   ├── vision_example.py  # Vision model examples
│   └── test_system.py     # System tests
├── tests/                 # Test files
│   └── notebook/
│       └── vlm_test.ipynb # Jupyter notebook examples
├── .env.example           # Environment template
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Configuration

#### Using BaseModelConfig

```python
from src.config.base_config import BaseModelConfig, Provider, ModelType

config = BaseModelConfig(
    model_name="anthropic/claude-3-haiku",
    provider=Provider.OPEN_ROUTER,
    model_type=ModelType.TEXT,
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=500
)
```

#### Using Provider-Specific Configs

```python
from src.config.provider_config import OpenRouterConfig

config = OpenRouterConfig(
    model_name="nvidia/nemotron-nano-12b-v2-vl:free",
    model_type=ModelType.MULTIMODAL,
    api_key="your-api-key"
)
# base_url is automatically set to OpenRouter's URL
```

### Image Search

```python
from src.core.agent.image_search import ImageSearch
from src.config.model_config import ImageSearchConfig

# Create config
search_config = ImageSearchConfig(
    model_name="google_image_search",
    provider=Provider.GOOGLE,
    api_key=os.getenv("GOOGLE_SEARCH_ENGINE_KEY"),
    search_engine_id=os.getenv("GOOGLE_ID_CSE")
)

# Search for images
searcher = ImageSearch(search_config)
image_urls = searcher.search("cats", num=5)
```

### Complete Workflow: Search + Analysis

```python
async def search_and_analyze(search_query: str, analysis_prompt: str):
    # Search for images
    image_urls = searcher.search(search_query, num=3)
    
    # Analyze with VLM
    for url in image_urls:
        response = await vlm_model.generate_multimodal(
            analysis_prompt,
            images=[url]
        )
        print(f"Image: {url}")
        print(f"Analysis: {response}\n")
```

## 🔧 Advanced Usage

### Adding New Providers

1. Create provider file in `src/providers/`
2. Implement required model classes
3. Register with factory
4. Add to registry

```python
# src/providers/custom_provider.py
from ..base.model import TextModel
from ..base.factory import ModelFactory

class CustomTextModel(TextModel):
    def __init__(self, config):
        super().__init__(config)
        # Your implementation
    
    async def generate_text(self, prompt, **kwargs):
        # Your implementation
        pass

# Auto-register
ModelFactory.register(Provider.CUSTOM, ModelType.TEXT, CustomTextModel)
```

### Using in Jupyter Notebooks

```python
# Setup PYTHONPATH
import sys
import os
sys.path.insert(0, "/path/to/god-eyes")

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import and use
from src.providers import registry
# ... rest of your code
```

## 🧪 Testing

```bash
# Run system tests
python examples/test_system.py

# Run specific examples
python examples/usage_example.py
python examples/vision_example.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Apache License 2.0](LICENSE)

## 🙏 Acknowledgments

- OpenRouter for unified API access
- Anthropic for Claude models
- OpenAI for GPT models
- Google for Custom Search API

## 📧 Contact

[sktkctman2@gmail.com](mailto:sktkctman2@gmail.com)

---