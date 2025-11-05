from ..base.model import TextModel, Provider, ModelType
from ..base.factory import ModelFactory
import anthropic

class AnthropicTextModel(TextModel):
    def __init__(self, model_name: str, provider: Provider = Provider.ANTHROPIC, **kwargs):
        super().__init__(model_name, provider, **kwargs)
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self.generate_text(prompt, **kwargs)

    async def generate_text(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        response = await self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.content[0].text

    async def generate_stream(self, prompt: str, **kwargs):
        async with self.client.messages.stream(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def validate_config(self) -> bool:
        return self.api_key is not None

# Auto-register model
ModelFactory.register(Provider.ANTHROPIC, ModelType.TEXT, AnthropicTextModel)