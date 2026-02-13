"""Registry and discovery helpers for graph-builder plugins."""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from pig.graphs.base import GraphBuilderFactory

_GRAPH_BUILDERS: dict[str, GraphBuilderFactory] = {}
_DISCOVERED = False


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def register_graph_builder(name: str):
    """Decorator to register a graph-builder factory by name."""

    normalized = _normalize_name(name)

    def decorator(factory: GraphBuilderFactory) -> GraphBuilderFactory:
        _GRAPH_BUILDERS[normalized] = factory
        return factory

    return decorator


def discover_graph_builders(force: bool = False) -> None:
    """Import all graph strategy modules in `pig.graphs`."""
    global _DISCOVERED
    if _DISCOVERED and not force:
        return

    import pig.graphs as graphs_pkg

    for module_info in pkgutil.iter_modules(graphs_pkg.__path__):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        if module_name in {"base", "registry"}:
            continue
        importlib.import_module(f"{graphs_pkg.__name__}.{module_name}")

    _DISCOVERED = True


def list_graph_builders() -> list[str]:
    """Return sorted names of all registered graph builders."""
    discover_graph_builders()
    return sorted(_GRAPH_BUILDERS)


def create_graph_builder(name: str, **kwargs: Any):
    """Instantiate a registered graph builder."""
    discover_graph_builders()
    normalized = _normalize_name(name)

    try:
        factory = _GRAPH_BUILDERS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_GRAPH_BUILDERS)) or "<none>"
        raise ValueError(
            f"Unknown graph builder '{name}'. Available builders: {available}"
        ) from exc

    return factory(**kwargs)
