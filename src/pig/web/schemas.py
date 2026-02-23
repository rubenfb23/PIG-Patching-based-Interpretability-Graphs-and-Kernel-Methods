"""Schema definitions for the local graph viewer websocket protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ViewFilter:
    """Client-selected 4D filter for graph projection."""

    slice_id: str
    layer_min: Optional[int] = None
    layer_max: Optional[int] = None
    token_min: Optional[int] = None
    token_max: Optional[int] = None
    min_abs_weight: float = 0.0
    max_edges: int = 1000


def build_message(message_type: str, **payload: Any) -> dict[str, Any]:
    """Build a typed websocket message envelope."""
    return {
        "type": message_type,
        **payload,
    }
