"""Model backends behind one interface."""

from nanoclaw.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ToolCallRequest,
    parse_tool_arguments,
)
from nanoclaw.providers.factory import provider_from_env
from nanoclaw.providers.fake import FakeProvider
from nanoclaw.providers.openai_compat import OpenAICompatProvider

__all__ = [
    "FakeProvider",
    "LLMProvider",
    "LLMResponse",
    "LLMUsage",
    "OpenAICompatProvider",
    "ToolCallRequest",
    "parse_tool_arguments",
    "provider_from_env",
]
