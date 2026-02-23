"""Data service for local interactive graph viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pig.graph import PatchInfluenceGraph, create_graph_builder
from pig.patching import PatchEffectDataset, PatchEffectTensor
from pig.prompts import SliceLabel
from pig.web.schemas import ViewFilter


@dataclass(frozen=True)
class ViewerConfig:
    """Configuration for graph preparation and response sizing."""

    top_k: int = 5
    enforce_direction: bool = True
    max_edges_default: int = 1000


def _slice_to_id(slice_label: SliceLabel) -> str:
    return f"{slice_label.task}/{slice_label.corruption}"


def _parse_slice_id(value: str) -> SliceLabel:
    task, corruption = value.split("/", maxsplit=1)
    return SliceLabel(task=task, corruption=corruption)


class GraphViewerService:
    """Prepare and filter graph data for the web viewer."""

    def __init__(
        self,
        dataset: PatchEffectDataset,
        config: ViewerConfig,
    ):
        if len(dataset) == 0:
            message = "Cannot initialize viewer service with empty dataset"
            raise ValueError(message)

        self._dataset = dataset
        self._config = config
        self._builder = create_graph_builder(
            k=config.top_k,
            enforce_direction=config.enforce_direction,
        )
        self._graphs = self._builder.build_all(dataset)
        self._slice_ids = sorted(_slice_to_id(label) for label in self._graphs)
        self._token_labels_by_slice = self._build_token_labels()

    @property
    def slice_ids(self) -> list[str]:
        return self._slice_ids

    def _build_token_labels(self) -> dict[str, list[str]]:
        """Build token label lists keyed by slice_id from the first tensor per slice."""
        labels: dict[str, list[str]] = {}
        for slice_label in self._graphs:
            slice_id = _slice_to_id(slice_label)
            tensors = self._dataset.get_by_slice(slice_label)
            if tensors and tensors[0].token_labels:
                graph = self._graphs[slice_label]
                labels[slice_id] = tensors[0].token_labels[: graph.num_tokens]
        return labels

    def initial_payload(self) -> dict:
        first_slice = self._slice_ids[0]
        first_graph = self.filtered_graph(
            ViewFilter(
                slice_id=first_slice,
                max_edges=self._config.max_edges_default,
            )
        )
        return {
            "slices": self._slice_ids,
            "default_filter": {
                "slice_id": first_slice,
                "min_abs_weight": 0.0,
                "max_edges": self._config.max_edges_default,
            },
            "graph": first_graph,
        }

    def filtered_graph(self, view_filter: ViewFilter) -> dict:
        slice_label = _parse_slice_id(view_filter.slice_id)
        graph = self._graphs.get(slice_label)
        if graph is None:
            raise ValueError(f"Unknown slice '{view_filter.slice_id}'")

        token_labels = self._token_labels_by_slice.get(view_filter.slice_id, [])
        return self._filter_graph(graph, view_filter, token_labels)

    def _filter_graph(
        self,
        graph: PatchInfluenceGraph,
        view_filter: ViewFilter,
        token_labels: list[str],
    ) -> dict:
        layer_min = 0 if view_filter.layer_min is None else view_filter.layer_min
        layer_max = (
            graph.num_layers - 1
            if view_filter.layer_max is None
            else view_filter.layer_max
        )
        token_min = 0 if view_filter.token_min is None else view_filter.token_min
        token_max = (
            graph.num_tokens - 1
            if view_filter.token_max is None
            else view_filter.token_max
        )

        selected_indices: set[int] = set()
        nodes = []
        for index, node in enumerate(graph.nodes):
            in_layer_range = layer_min <= node.layer <= layer_max
            in_token_range = token_min <= node.token <= token_max
            if in_layer_range and in_token_range:
                selected_indices.add(index)
                nodes.append(
                    {
                        "id": index,
                        "layer": node.layer,
                        "token": node.token,
                        "type": node.node_type,
                        "head": node.head,
                    }
                )

        min_abs_weight = max(
            0.0,
            view_filter.min_abs_weight,
        )
        edge_candidates = []
        for edge in graph.edges:
            if edge.src not in selected_indices or edge.dst not in selected_indices:
                continue
            if abs(edge.weight) < min_abs_weight:
                continue
            edge_candidates.append(edge)

        edge_candidates.sort(key=lambda edge: abs(edge.weight), reverse=True)
        limited_edges = edge_candidates[: max(0, view_filter.max_edges)]

        edges = [
            {
                "src": edge.src,
                "dst": edge.dst,
                "weight": edge.weight,
            }
            for edge in limited_edges
        ]

        return {
            "slice": view_filter.slice_id,
            "num_layers": graph.num_layers,
            "num_tokens": graph.num_tokens,
            "token_labels": token_labels,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "selected_nodes": len(nodes),
                "selected_edges": len(edges),
                "graph_nodes": graph.num_nodes,
                "graph_edges": graph.num_edges,
            },
        }


def load_dataset_from_cache(cache_dir: Path | str) -> PatchEffectDataset:
    """Load patch tensors from JSON cache files into a dataset."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise ValueError(f"Cache directory does not exist: {cache_path}")

    cache_files = sorted(cache_path.glob("*.json"))
    if not cache_files:
        raise ValueError(f"No cache files found in: {cache_path}")

    tensors: list[PatchEffectTensor] = []
    for file_path in cache_files:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        tensors.append(PatchEffectTensor.from_dict(payload))

    grouped: dict[tuple[tuple[str, int | None], ...], list[PatchEffectTensor]] = {}
    for tensor in tensors:
        axis_key = tuple(
            (component.node_type, component.head) for component in tensor.component_axis
        )
        grouped.setdefault(axis_key, []).append(tensor)

    selected_key = max(
        grouped.keys(),
        key=lambda key: (len(key), len(grouped[key])),
    )
    selected_tensors = grouped[selected_key]

    dropped = len(tensors) - len(selected_tensors)
    if dropped > 0:
        print(
            "Viewer cache loader: skipped "
            f"{dropped} tensors with incompatible component_axis; "
            f"using axis size {len(selected_key)}"
        )

    dataset = PatchEffectDataset()
    for tensor in selected_tensors:
        dataset.add(tensor)

    return dataset
