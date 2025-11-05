from ..base.model import TextModel, MultiModalModel
from ..base.factory import ModelFactory
from ..config.base_config import BaseModelConfig, Provider, ModelType
from openai import AsyncOpenAI

class OpenRouterTextModel(TextModel):
    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        if not self.api_key:
            raise ValueError("API key is required for OpenRouter")
        self.client = AsyncOpenAI(
            base_url=self.base_url or "https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_text(prompt, **kwargs)
    
    async def generate_text(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return response.choices[0].message.content
    
    async def generate_stream(self, prompt: str, **kwargs):
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=kwargs.get('temperature', self.config.temperature),
            **kwargs
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def validate_config(self) -> bool:
        return self.api_key is not None

class OpenRouterMultiModalModel(MultiModalModel):
    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        if not self.api_key:
            raise ValueError("API key is required for OpenRouter")
        self.client = AsyncOpenAI(
            base_url=self.base_url or "https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_multimodal(prompt, **kwargs)

    async def generate_multimodal(self, text: str, images=None, **kwargs) -> str:
        messages = [{"role": "user", "content": text}]
        
        if images:
            content = [{"type": "text", "text": text}]
            for image in images:
                if isinstance(image, str):  # URL
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image}
                    })
            messages = [{"role": "user", "content": content}]
        
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
            temperature=kwargs.get('temperature', self.config.temperature),
            **kwargs
        )
        return response.choices[0].message.content

    def validate_config(self) -> bool:
        return self.api_key is not None

# Auto-register models when module is imported
ModelFactory.register(Provider.OPEN_ROUTER, ModelType.TEXT, OpenRouterTextModel)
ModelFactory.register(Provider.OPEN_ROUTER, ModelType.MULTIMODAL, OpenRouterMultiModalModel)