"""Shared protocols and type aliases for graph-builder plugins."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from pig.patching import PatchEffectDataset
from pig.prompts import SliceLabel


@runtime_checkable
class GraphBuilderProtocol(Protocol):
    """Protocol implemented by graph-builder strategies."""

    def build_from_slice(self, dataset: PatchEffectDataset, slice_label: SliceLabel):
        ...

    def build_all(self, dataset: PatchEffectDataset):
        ...

    def build_per_example(self, dataset: PatchEffectDataset):
        ...


GraphBuilderFactory = Callable[..., GraphBuilderProtocol]
