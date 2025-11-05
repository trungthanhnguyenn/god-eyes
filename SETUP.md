# God Eyes Setup Guide 🚀

This guide will help you get God Eyes up and running in under 5 minutes!

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning)

## 🔧 Step-by-Step Setup

### Step 1: Clone or Download

```bash
git clone https://github.com/yourusername/god-eyes.git
cd god-eyes
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Get API Keys

#### OpenRouter (Required for VLM)

1. Go to [https://openrouter.ai/](https://openrouter.ai/)
2. Sign up or log in
3. Go to "Keys" section
4. Create a new API key
5. Copy the key (starts with `sk-or-v1-...`)

#### Google Custom Search (Required for Image Search)

**Google API Key:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable "Custom Search API"
4. Go to "Credentials" → "Create Credentials" → "API Key"
5. Copy the API key (starts with `AIza...`)

**Search Engine ID:**
1. Go to [Programmable Search Engine](https://programmablesearchengine.google.com/)
2. Click "Add" to create a new search engine
3. Set "Sites to search" to "Search the entire web"
4. Enable "Image search" in settings
5. Copy the "Search engine ID" (cx parameter)

### Step 5: Configure Environment

```bash
# Copy example file
cp .env.example .env

# Edit with your favorite editor
nano .env  # or vim, code, etc.
```

**Minimal .env configuration:**
```properties
OPENROUTER_API_KEY=sk-or-v1-your-key-here
LLM_MODEL_NAME=minimax/minimax-m2:free
VLM_MODEL_NAME=nvidia/nemotron-nano-12b-v2-vl:free
GOOGLE_SEARCH_ENGINE_KEY=AIzaSy-your-key-here
GOOGLE_ID_CSE=your-search-engine-id
```

### Step 6: Test Installation

```bash
# Run system test
python examples/test_system.py

# Run usage example
python examples/usage_example.py
```

### Step 7: Try Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open tests/notebook/vlm_test.ipynb
# Run cells to test VLM functionality
```

## 🎯 Quick Test Script

Create a file `test_quick.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from src.base.factory import ModelFactory
from src.config.base_config import BaseModelConfig, Provider, ModelType

load_dotenv()
from src.providers import registry

async def main():
    print("Testing God Eyes...")
    
    # Test 1: Text Model
    print("\nTesting Text Model...")
    config = BaseModelConfig(
        model_name="minimax/minimax-m2:free",
        provider=Provider.OPEN_ROUTER,
        model_type=ModelType.TEXT,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7
    )
    
    model = ModelFactory.create_model(config)
    response = await model.generate("Say hello!")
    print(f"Response: {response}")
    
    # Test 2: Vision Model
    print("\nTesting Vision Model...")
    vlm_config = BaseModelConfig(
        model_name=os.getenv("VLM_MODEL_NAME"),
        provider=Provider.OPEN_ROUTER,
        model_type=ModelType.MULTIMODAL,
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    vlm = ModelFactory.create_model(vlm_config)
    if hasattr(vlm, 'generate_multimodal'):
        response = await vlm.generate_multimodal(
            "Describe this sunset",
            images=["https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Sunset_at_the_Golden_Gate_Bridge.jpg/320px-Sunset_at_the_Golden_Gate_Bridge.jpg"]
        )
        print(f"Response: {response}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python test_quick.py
```

## 🐛 Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```python
import sys
sys.path.insert(0, '/absolute/path/to/god-eyes')
```

### API Key Errors

**Problem:** "API key not valid"

**Solution:**
- Check for extra spaces or quotes in .env
- Make sure key starts with correct prefix (sk-or-v1- for OpenRouter)
- Reload .env: `load_dotenv(override=True)`

### Google Search Errors

**Problem:** "API key not valid" for Google

**Solution:**
- Enable Custom Search API in Google Cloud Console
- Make sure API key starts with `AIza` (not `AAIza`)
- Check Search Engine ID is correct

### Async Errors in Jupyter

**Problem:** "RuntimeError: asyncio.run() cannot be called from a running event loop"

**Solution:**
Use `await` instead of `asyncio.run()` in Jupyter notebooks:
```python
# Don't use this in Jupyter
# asyncio.run(my_function())

# Use this instead
await my_function()
```

## 📚 Next Steps

1. Read the [full README](README.md)
2. Explore [examples/](examples/)
3. Try the [Jupyter notebook](tests/notebook/vlm_test.ipynb)
4. Check out provider-specific docs in [docs/](docs/)

## 💬 Need Help?

- Check [Issues](https://github.com/trungthanhnguyenn/god-eyes/issues)
- Contact: [sktkctman2@gmail.com](mailto:sktkctman2@gmail.com

---

Have fun!
