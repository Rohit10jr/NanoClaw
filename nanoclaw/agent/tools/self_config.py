"""The ``my`` tool's config (nanobot agent/tools/self_config.py).

NanoClaw keeps every config model in ``nanoclaw.config.schema`` (the schema
imports nothing from the agent); this module keeps nanobot's import path.
"""

from nanoclaw.config.schema import MyToolConfig

__all__ = ["MyToolConfig"]
