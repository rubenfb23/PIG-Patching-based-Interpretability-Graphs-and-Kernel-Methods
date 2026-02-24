"""Graph strategy using correlation over fine-tune minus base effects."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from pig.graph import Edge, GraphBuilder, PatchInfluenceGraph
from pig.graphs.registry import register_graph_builder
from pig.patching import PatchEffectDataset, PatchEffectTensor
from pig.prompts import SliceLabel

logger = logging.getLogger(__name__)


class DiffCorrelationGraphBuilder(GraphBuilder):
    """Build graphs from correlations of per-example diff effect profiles."""

    def __init__(
        self,
        base_dataset: PatchEffectDataset,
        k: int = 5,
        enforce_direction: bool = True,
        min_weight: float = 0.0,
    ):
        super().__init__(
            k=k,
            enforce_direction=enforce_direction,
            min_weight=min_weight,
        )
        self._base_dataset = base_dataset

    @staticmethod
    def _prompt_key(tensor: PatchEffectTensor) -> tuple[str, str, str]:
        pair = tensor.prompt_pair
        return (pair.x_cln, pair.x_crp, pair.y_star)

    def build_from_slice(
        self,
        dataset: PatchEffectDataset,
        slice_label: SliceLabel,
    ) -> PatchInfluenceGraph:
        ft_tensors = dataset.get_by_slice(slice_label)
        if not ft_tensors:
            raise ValueError(f"No tensors found for slice {slice_label}")

        base_tensors = self._base_dataset.get_by_slice(slice_label)
        if not base_tensors:
            raise ValueError(f"No base tensors found for slice {slice_label}")

        base_by_key = {self._prompt_key(tensor): tensor for tensor in base_tensors}

        num_layers = ft_tensors[0].num_layers
        component_axis = dataset.get_component_axis()

        matched_pairs: list[tuple[PatchEffectTensor, PatchEffectTensor]] = []
        for ft_tensor in ft_tensors:
            key = self._prompt_key(ft_tensor)
            base_tensor = base_by_key.get(key)
            if base_tensor is None:
                logger.info(
                    "Skipping unmatched prompt pair for slice %s: %s",
                    slice_label,
                    key,
                )
                continue

            shapes_compatible = (
                base_tensor.num_layers == ft_tensor.num_layers
                and base_tensor.num_components == ft_tensor.num_components
                and base_tensor.component_axis == ft_tensor.component_axis
            )
            if not shapes_compatible:
                logger.warning(
                    "Skipping incompatible tensor pair for slice %s (ft shape=%s, "
                    "base shape=%s)",
                    slice_label,
                    ft_tensor.shape,
                    base_tensor.shape,
                )
                continue

            matched_pairs.append((ft_tensor, base_tensor))

        if not matched_pairs:
            raise ValueError(
                f"No matched prompt pairs for slice {slice_label}"
            )

        num_tokens = min(
            min(ft_tensor.num_tokens for ft_tensor, _ in matched_pairs),
            min(base_tensor.num_tokens for _, base_tensor in matched_pairs),
        )

        diff_rows: list[NDArray[np.float32]] = []
        for ft_tensor, base_tensor in matched_pairs:
            diff = (
                ft_tensor.effects[:, :num_tokens, :]
                - base_tensor.effects[:, :num_tokens, :]
            )
            diff_rows.append(diff.reshape(-1))

        effect_matrix = np.stack(diff_rows, axis=0).astype(np.float32)
        corr_matrix = self._compute_correlation_matrix(effect_matrix)

        nodes = self._create_nodes(num_layers, num_tokens, component_axis)
        if self.enforce_direction:
            corr_matrix = self._apply_direction_constraint(corr_matrix, nodes)

        sparse_matrix = self._apply_topk_sparsification(corr_matrix)

        edges: list[Edge] = []
        for src_idx in range(len(nodes)):
            for dst_idx in range(len(nodes)):
                weight = sparse_matrix[src_idx, dst_idx]
                if abs(weight) > self.min_weight and src_idx != dst_idx:
                    edges.append(Edge(src=src_idx, dst=dst_idx, weight=float(weight)))

        return PatchInfluenceGraph(
            nodes=nodes,
            edges=edges,
            slice_label=slice_label,
            num_layers=num_layers,
            num_tokens=num_tokens,
            metadata={
                "k": self.k,
                "enforce_direction": self.enforce_direction,
                "num_examples": len(matched_pairs),
                "graph_builder": self.__class__.__name__,
                "mode": "diff_correlation",
            },
        )

    def build_per_example(
        self,
        dataset: PatchEffectDataset,
    ) -> list[tuple[PatchInfluenceGraph, SliceLabel]]:
        _ = dataset
        raise NotImplementedError(
            "DiffCorrelationGraphBuilder no soporta build_per_example"
        )


@register_graph_builder("diff_correlation_topk")
def create_builder(**kwargs):
    """Create a diff-correlation top-k graph builder."""
    base_dataset = kwargs.pop("base_dataset", None)
    if base_dataset is None:
        raise ValueError("'diff_correlation_topk' requiere 'base_dataset'")
    return DiffCorrelationGraphBuilder(base_dataset=base_dataset, **kwargs)
