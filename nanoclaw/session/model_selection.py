"""Session-scoped model preset metadata (nanobot session/model_selection.py)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

# nanobot's key, so a session file keeps its /model choice in either project.
SESSION_MODEL_PRESET_METADATA_KEY = "_nanobot_model_preset"


def model_preset_from_metadata(metadata: object) -> str | None:
    """The session's preset name from its saved metadata, or None."""
    if not isinstance(metadata, Mapping):
        return None
    typed_metadata = cast(Mapping[object, object], metadata)
    if SESSION_MODEL_PRESET_METADATA_KEY not in typed_metadata:
        return None
    value = typed_metadata[SESSION_MODEL_PRESET_METADATA_KEY]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("session model preset must be a non-empty string")
    return value.strip()
