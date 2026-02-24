"""Unit tests for diff-correlation graph builder and viewer integration."""

from __future__ import annotations

import numpy as np
import pytest

from pig.graph import Node, PatchInfluenceGraph, create_graph_builder
from pig.graphs.diff_correlation_topk import DiffCorrelationGraphBuilder
from pig.graphs.registry import list_graph_builders
from pig.patching import ComponentSpec, PatchEffectDataset, PatchEffectTensor
from pig.prompts import PromptPair, SliceLabel
from pig.web.schemas import ViewFilter
from pig.web.service import GraphViewerService, ViewerConfig

_COMPONENT_AXIS = [ComponentSpec(node_type="res")]
_SLICE = SliceLabel(task="ioi", corruption="name_swap")
_SLICE_EXTRA = SliceLabel(task="ioi", corruption="abba")


def _pair(index: int, slice_label: SliceLabel) -> PromptPair:
    return PromptPair(
        x_cln=f"clean-{index}",
        x_crp=f"corrupt-{index}",
        y_star="target",
        slice_label=slice_label,
        meta={},
    )


def _tensor(prompt_pair: PromptPair, effects: np.ndarray) -> PatchEffectTensor:
    return PatchEffectTensor(
        effects=effects.astype(np.float32),
        component_axis=_COMPONENT_AXIS,
        prompt_pair=prompt_pair,
        base_score=-1.0,
        clean_score=1.0,
    )


def _build_aligned_dataset_pair(
    count: int = 8,
) -> tuple[PatchEffectDataset, PatchEffectDataset]:
    rng = np.random.default_rng(1234)
    base_dataset = PatchEffectDataset()
    ft_dataset = PatchEffectDataset()

    for idx in range(count):
        prompt_pair = _pair(idx, _SLICE)
        base = rng.normal(0.0, 0.05, size=(2, 2, 1)).astype(np.float32)

        z = np.float32(idx - 3.5)
        delta = np.zeros((2, 2, 1), dtype=np.float32)
        delta[0, 0, 0] = z
        delta[1, 0, 0] = np.float32(2.0 * z + rng.normal(0.0, 0.01))
        delta[0, 1, 0] = np.float32(rng.normal(0.0, 0.02))
        delta[1, 1, 0] = np.float32(rng.normal(0.0, 0.02))

        ft = base + delta
        base_dataset.add(_tensor(prompt_pair, base))
        ft_dataset.add(_tensor(prompt_pair, ft))

    return base_dataset, ft_dataset


@pytest.fixture
def aligned_datasets() -> tuple[PatchEffectDataset, PatchEffectDataset]:
    return _build_aligned_dataset_pair(count=8)


@pytest.fixture
def partial_overlap_datasets() -> tuple[PatchEffectDataset, PatchEffectDataset]:
    base_dataset, ft_dataset = _build_aligned_dataset_pair(count=8)
    unmatched_prompt = _pair(999, _SLICE)
    unmatched_effects = np.ones((2, 2, 1), dtype=np.float32)
    ft_dataset.add(_tensor(unmatched_prompt, unmatched_effects))
    return base_dataset, ft_dataset


def test_build_from_slice_returns_graph(aligned_datasets) -> None:
    base_dataset, ft_dataset = aligned_datasets
    builder = DiffCorrelationGraphBuilder(base_dataset=base_dataset, k=2)

    graph = builder.build_from_slice(ft_dataset, _SLICE)

    assert isinstance(graph, PatchInfluenceGraph)
    assert graph.num_layers == 2
    assert graph.num_tokens == 2
    assert graph.num_nodes == 4


def test_diff_metadata_records_matched_count(aligned_datasets) -> None:
    base_dataset, ft_dataset = aligned_datasets
    builder = DiffCorrelationGraphBuilder(base_dataset=base_dataset, k=2)

    graph = builder.build_from_slice(ft_dataset, _SLICE)

    assert graph.metadata["num_examples"] == 8
    assert graph.metadata["mode"] == "diff_correlation"


def test_partial_overlap_skips_unmatched(partial_overlap_datasets) -> None:
    base_dataset, ft_dataset = partial_overlap_datasets
    builder = DiffCorrelationGraphBuilder(base_dataset=base_dataset, k=2)

    graph = builder.build_from_slice(ft_dataset, _SLICE)

    assert graph.metadata["num_examples"] == 8


def test_raises_when_no_matching_pairs() -> None:
    base_dataset = PatchEffectDataset()
    ft_dataset = PatchEffectDataset()

    for idx in range(4):
        base_dataset.add(
            _tensor(_pair(idx, _SLICE), np.zeros((2, 2, 1), dtype=np.float32))
        )
        ft_dataset.add(
            _tensor(_pair(idx + 100, _SLICE), np.ones((2, 2, 1), dtype=np.float32))
        )

    builder = DiffCorrelationGraphBuilder(base_dataset=base_dataset, k=2)
    with pytest.raises(ValueError, match="No matched prompt pairs"):
        builder.build_from_slice(ft_dataset, _SLICE)


def test_raises_when_slice_missing_in_base(aligned_datasets) -> None:
    _, ft_dataset = aligned_datasets
    base_dataset = PatchEffectDataset()
    builder = DiffCorrelationGraphBuilder(base_dataset=base_dataset, k=2)

    with pytest.raises(ValueError, match="No base tensors found"):
        builder.build_from_slice(ft_dataset, _SLICE)


def test_direction_constraint_respected(aligned_datasets) -> None:
    base_dataset, ft_dataset = aligned_datasets
    builder = DiffCorrelationGraphBuilder(
        base_dataset=base_dataset,
        k=4,
        enforce_direction=True,
    )

    graph = builder.build_from_slice(ft_dataset, _SLICE)

    for edge in graph.edges:
        assert graph.nodes[edge.src] < graph.nodes[edge.dst]


def test_build_all_iterates_ft_slices() -> None:
    base_dataset, ft_dataset = _build_aligned_dataset_pair(count=6)
    for idx in range(3):
        pair = _pair(200 + idx, _SLICE_EXTRA)
        base_effects = np.full((2, 2, 1), idx, dtype=np.float32)
        ft_effects = base_effects + 0.1
        base_dataset.add(_tensor(pair, base_effects))
        ft_dataset.add(_tensor(pair, ft_effects))

    builder = DiffCorrelationGraphBuilder(base_dataset=base_dataset, k=2)
    graphs = builder.build_all(ft_dataset)

    assert set(graphs.keys()) == set(ft_dataset.get_slices())


def test_known_diff_structure_produces_correlation(aligned_datasets) -> None:
    base_dataset, ft_dataset = aligned_datasets
    builder = DiffCorrelationGraphBuilder(base_dataset=base_dataset, k=2)

    graph = builder.build_from_slice(ft_dataset, _SLICE)

    src_idx = Node(layer=0, token=0).to_index(graph.num_tokens, _COMPONENT_AXIS)
    dst_idx = Node(layer=1, token=0).to_index(graph.num_tokens, _COMPONENT_AXIS)
    matching_edges = [
        edge for edge in graph.edges if edge.src == src_idx and edge.dst == dst_idx
    ]
    assert matching_edges
    assert matching_edges[0].weight > 0.8


def test_registry_name_is_discoverable() -> None:
    names = list_graph_builders()
    assert "diff_correlation_topk" in names


def test_registry_factory_rejects_missing_base_dataset() -> None:
    with pytest.raises(ValueError, match="requiere 'base_dataset'"):
        create_graph_builder("diff_correlation_topk", k=2)


def test_service_accepts_injected_builder(aligned_datasets) -> None:
    base_dataset, ft_dataset = aligned_datasets
    builder = DiffCorrelationGraphBuilder(base_dataset=base_dataset, k=2)
    service = GraphViewerService(
        dataset=ft_dataset,
        config=ViewerConfig(top_k=2),
        builder=builder,
    )

    payload = service.initial_payload()

    assert payload["slices"]
    assert payload["graph"]["stats"]["selected_nodes"] > 0


def test_service_default_builder_unchanged(aligned_datasets) -> None:
    _, ft_dataset = aligned_datasets
    service = GraphViewerService(
        dataset=ft_dataset,
        config=ViewerConfig(top_k=2),
    )

    payload = service.filtered_graph(
        ViewFilter(slice_id="ioi/name_swap", max_edges=20)
    )
    assert payload["stats"]["graph_nodes"] > 0
    assert payload["stats"]["graph_edges"] > 0


def test_service_skips_slices_with_no_overlap(aligned_datasets) -> None:
    base_dataset, ft_dataset = aligned_datasets
    for idx in range(3):
        pair = _pair(400 + idx, _SLICE_EXTRA)
        ft_dataset.add(_tensor(pair, np.full((2, 2, 1), 3.0, dtype=np.float32)))

    builder = DiffCorrelationGraphBuilder(base_dataset=base_dataset, k=2)
    service = GraphViewerService(
        dataset=ft_dataset,
        config=ViewerConfig(top_k=2),
        builder=builder,
    )

    assert service.slice_ids == ["ioi/name_swap"]
    with pytest.raises(ValueError, match="Unknown slice"):
        service.filtered_graph(ViewFilter(slice_id="ioi/abba"))
