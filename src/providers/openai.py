from ..base.model import TextModel, MultiModalModel, Provider, ModelType
from ..base.factory import ModelFactory
import openai

class OpenAITextModel(TextModel):
    def __init__(self, model_name: str, provider: Provider = Provider.OPENAI, **kwargs):
        super().__init__(model_name, provider, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_text(prompt, **kwargs)

    async def generate_text(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    async def generate_stream(self, prompt: str, **kwargs):
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def validate_config(self) -> bool:
        return self.api_key is not None

class OpenAIVisionModel(MultiModalModel):
    def __init__(self, model_name: str, provider: Provider = Provider.OPENAI, **kwargs):
        super().__init__(model_name, provider, **kwargs)
        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_multimodal(prompt, **kwargs)

    async def generate_multimodal(self, text: str, images=None, **kwargs) -> str:
        messages = [{"role": "user", "content": text}]
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content

    def validate_config(self) -> bool:
        return self.api_key is not None

# Auto-register models when module is imported
ModelFactory.register(Provider.OPENAI, ModelType.TEXT, OpenAITextModel)
ModelFactory.register(Provider.OPENAI, ModelType.MULTIMODAL, OpenAIVisionModel)