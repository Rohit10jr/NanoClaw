"""Shared pydantic base model for configuration models.

Mirrors nanobot ``config_base.py``: JSON uses camelCase (``maxTokens``),
Python uses snake_case (``max_tokens``), and both spellings are accepted on
input. Unknown keys are ignored (pydantic's default, as in nanobot), so a full
nanobot ``config.json`` loads: NanoClaw reads the settings it supports and
skips the rest.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class Base(BaseModel):
    """Base model that accepts both camelCase and snake_case keys."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)
