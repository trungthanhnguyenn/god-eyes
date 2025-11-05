import os
from dotenv import load_dotenv

from src.base.factory import ModelFactory
from src.config.base_config import Provider, ModelType
from src.config.provider_config import OpenRouterConfig

# Load environment variables
load_dotenv()

# Import registry to auto-register all providers
from src.providers import registry

class PlanerAgent:
    """Agent that plans execution"""

    def __init__(self):
        # Initialize the planner model from OpenRouter
        config = OpenRouterConfig(
            model_name=os.getenv("LLM_MODEL_NAME", ""),  # Example model name
            api_key=os.getenv("OPENROUTER_API_KEY"),
            provider=Provider.OPEN_ROUTER,
            model_type=ModelType.TEXT,
            max_tokens=1000,
            temperature=0.7
        )
        self.planner_model = ModelFactory.create_model(config)

    
    def build_prompt(self, user_input: str, prompt_path: str) -> str:
        """Build the prompt for the planner model."""
        with open(prompt_path, 'r') as file:
            prompt_template = file.read()
        return prompt_template.replace("{{user_input}}", user_input)

    async def plan(self, prompt: str):
        """Generate plan based on the given prompt using the planner model."""
        if not self.planner_model.is_available:
            raise RuntimeError("Planner model is not available")

        response = await self.planner_model.generate_text(prompt)
        return response