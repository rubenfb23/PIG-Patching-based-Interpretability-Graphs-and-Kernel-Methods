"""Weisfeiler-Lehman (WL) graph embeddings.

This module implements WL subtree feature extraction for converting
patch-influence graphs into fixed-dimensional feature vectors suitable
for kernel methods.

The WL algorithm iteratively refines node labels by hashing neighborhood
structures, capturing increasingly complex subgraph patterns.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from pig.graph import PatchInfluenceGraph
from pig.prompts import SliceLabel


def _hash_label(label: str) -> str:
    """Hash a label string to a shorter fixed-length string."""
    return hashlib.md5(label.encode()).hexdigest()[:8]


@dataclass
class WLEmbedding:
    """WL feature embedding for a single graph.

    Attributes:
        features: Dictionary mapping feature labels to counts
        graph_id: Identifier for the source graph
        depth: WL iteration depth used
    """

    features: dict[str, int]
    graph_id: str
    depth: int

    def to_vector(self, vocabulary: list[str]) -> NDArray[np.float32]:
        """Convert to dense vector using a vocabulary.

        Args:
            vocabulary: Ordered list of feature labels

        Returns:
            Dense feature vector of shape [len(vocabulary)]
        """
        vec = np.zeros(len(vocabulary), dtype=np.float32)
        for i, label in enumerate(vocabulary):
            vec[i] = self.features.get(label, 0)
        return vec

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "features": self.features,
            "graph_id": self.graph_id,
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WLEmbedding":
        """Deserialize from dictionary."""
        return cls(
            features=data["features"],
            graph_id=data["graph_id"],
            depth=data["depth"],
        )


class WLEncoder:
    """Weisfeiler-Lehman subtree feature encoder.

    Computes WL subtree features by iteratively refining node labels
    based on neighborhood structure.
    """

    def __init__(self, depth: int = 3, use_edge_weights: bool = True):
        """Initialize the encoder.

        Args:
            depth: Number of WL iterations (higher = more complex patterns)
            use_edge_weights: Whether to include edge weights in hashing
        """
        self.depth = depth
        self.use_edge_weights = use_edge_weights

    def _get_initial_labels(self, graph: PatchInfluenceGraph) -> list[str]:
        """Get initial node labels based on node attributes."""
        labels = []
        for node in graph.nodes:
            # Initial label encodes node position
            label = f"L{node.layer}_T{node.token}_{node.node_type}"
            if node.head is not None:
                label += f"_H{node.head}"
            labels.append(label)
        return labels

    def _get_neighbors(
        self, graph: PatchInfluenceGraph, node_idx: int
    ) -> list[tuple[int, float]]:
        """Get neighbors and edge weights for a node."""
        neighbors = []
        for edge in graph.edges:
            if edge.src == node_idx:
                neighbors.append((edge.dst, edge.weight))
            # For undirected version, could also check edge.dst == node_idx
        return neighbors

    def _refine_labels(
        self,
        graph: PatchInfluenceGraph,
        current_labels: list[str],
    ) -> list[str]:
        """Perform one WL refinement iteration."""
        new_labels = []

        for node_idx in range(len(graph.nodes)):
            # Get current label
            node_label = current_labels[node_idx]

            # Get neighbor labels (with optional weights)
            neighbors = self._get_neighbors(graph, node_idx)
            neighbor_labels = []

            for neighbor_idx, weight in neighbors:
                neighbor_label = current_labels[neighbor_idx]
                if self.use_edge_weights:
                    # Discretize weight for hashing
                    weight_bin = int(weight * 10)
                    neighbor_labels.append(f"{neighbor_label}:{weight_bin}")
                else:
                    neighbor_labels.append(neighbor_label)

            # Sort for determinism
            neighbor_labels.sort()

            # Create new label by hashing current + neighbors
            combined = f"{node_label}|{'_'.join(neighbor_labels)}"
            new_label = _hash_label(combined)
            new_labels.append(new_label)

        return new_labels

    def encode(self, graph: PatchInfluenceGraph) -> WLEmbedding:
        """Compute WL features for a graph.

        Args:
            graph: The graph to encode

        Returns:
            WLEmbedding containing feature counts
        """
        all_features: Counter[str] = Counter()

        # Initialize labels
        labels = self._get_initial_labels(graph)

        # Add initial labels as features (depth 0)
        for label in labels:
            all_features[f"d0_{label}"] += 1

        # Iteratively refine labels
        for d in range(1, self.depth + 1):
            labels = self._refine_labels(graph, labels)

            # Add refined labels as features
            for label in labels:
                all_features[f"d{d}_{label}"] += 1

        return WLEmbedding(
            features=dict(all_features),
            graph_id=str(graph.slice_label),
            depth=self.depth,
        )

    def encode_batch(
        self, graphs: list[PatchInfluenceGraph]
    ) -> list[WLEmbedding]:
        """Encode multiple graphs."""
        return [self.encode(g) for g in graphs]


@dataclass
class WLFeatureMatrix:
    """Feature matrix from WL encoding of multiple graphs.

    Provides a fixed-dimensional representation suitable for kernel methods.
    """

    embeddings: list[WLEmbedding]
    vocabulary: list[str]
    slice_labels: list[SliceLabel]

    @property
    def num_graphs(self) -> int:
        return len(self.embeddings)

    @property
    def num_features(self) -> int:
        return len(self.vocabulary)

    def to_matrix(self) -> NDArray[np.float32]:
        """Convert to dense feature matrix.

        Returns:
            Matrix of shape [num_graphs, num_features]
        """
        matrix = np.zeros(
            (self.num_graphs, self.num_features), dtype=np.float32
        )
        for i, emb in enumerate(self.embeddings):
            matrix[i] = emb.to_vector(self.vocabulary)
        return matrix

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "embeddings": [e.to_dict() for e in self.embeddings],
            "vocabulary": self.vocabulary,
            "slice_labels": [
                {"task": s.task, "corruption": s.corruption}
                for s in self.slice_labels
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WLFeatureMatrix":
        """Deserialize from dictionary."""
        embeddings = [WLEmbedding.from_dict(e) for e in data["embeddings"]]
        slice_labels = [
            SliceLabel(task=s["task"], corruption=s["corruption"])
            for s in data["slice_labels"]
        ]
        return cls(
            embeddings=embeddings,
            vocabulary=data["vocabulary"],
            slice_labels=slice_labels,
        )


class WLEmbeddingCache:
    """Cache for WL embeddings."""

    def __init__(self, cache_dir: Path | str = ".cache/wl_embeddings"):
        """Initialize the cache.

        Args:
            cache_dir: Directory for storing cached embeddings
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(
        self, graphs: list[PatchInfluenceGraph], depth: int
    ) -> str:
        """Compute cache key from graphs and parameters."""
        # Hash graph structure and parameters
        content = json.dumps({
            "slices": [str(g.slice_label) for g in graphs],
            "num_nodes": [g.num_nodes for g in graphs],
            "num_edges": [g.num_edges for g in graphs],
            "depth": depth,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(
        self, graphs: list[PatchInfluenceGraph], depth: int
    ) -> Optional[WLFeatureMatrix]:
        """Retrieve cached feature matrix."""
        key = self._compute_key(graphs, depth)
        cache_path = self.cache_dir / f"{key}.json"

        if cache_path.exists():
            with open(cache_path, "r") as f:
                data = json.load(f)
            return WLFeatureMatrix.from_dict(data)
        return None

    def put(
        self,
        feature_matrix: WLFeatureMatrix,
        graphs: list[PatchInfluenceGraph],
        depth: int,
    ) -> None:
        """Store feature matrix in cache."""
        key = self._compute_key(graphs, depth)
        cache_path = self.cache_dir / f"{key}.json"

        with open(cache_path, "w") as f:
            json.dump(feature_matrix.to_dict(), f)


def compute_wl_features(
    graphs: dict[SliceLabel, PatchInfluenceGraph],
    depth: int = 3,
    cache_dir: Optional[str] = None,
) -> WLFeatureMatrix:
    """Compute WL features for a collection of graphs.

    Args:
        graphs: Dictionary mapping slice labels to graphs
        depth: WL iteration depth
        cache_dir: Optional cache directory

    Returns:
        WLFeatureMatrix with embeddings for all graphs
    """
    graph_list = list(graphs.values())
    slice_labels = list(graphs.keys())

    # Check cache
    if cache_dir:
        cache = WLEmbeddingCache(cache_dir)
        cached = cache.get(graph_list, depth)
        if cached is not None:
            return cached

    # Compute embeddings
    encoder = WLEncoder(depth=depth)
    embeddings = encoder.encode_batch(graph_list)

    # Build vocabulary from all embeddings
    all_features: set[str] = set()
    for emb in embeddings:
        all_features.update(emb.features.keys())
    vocabulary = sorted(all_features)

    feature_matrix = WLFeatureMatrix(
        embeddings=embeddings,
        vocabulary=vocabulary,
        slice_labels=slice_labels,
    )

    # Cache result
    if cache_dir:
        cache.put(feature_matrix, graph_list, depth)

    return feature_matrix


def compute_wl_features_from_list(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    depth: int = 3,
    cache_dir: Optional[str] = None,
) -> WLFeatureMatrix:
    """Compute WL features from a list of (graph, label) tuples.

    This is useful for per-example graphs where each example has its own graph.

    Args:
        graphs_with_labels: List of (graph, slice_label) tuples
        depth: WL iteration depth
        cache_dir: Optional cache directory

    Returns:
        WLFeatureMatrix with embeddings for all graphs
    """
    graph_list = [g for g, _ in graphs_with_labels]
    slice_labels = [l for _, l in graphs_with_labels]

    # Check cache
    if cache_dir:
        cache = WLEmbeddingCache(cache_dir)
        cached = cache.get(graph_list, depth)
        if cached is not None:
            return cached

    # Compute embeddings
    encoder = WLEncoder(depth=depth)
    embeddings = encoder.encode_batch(graph_list)

    # Build vocabulary from all embeddings
    all_features: set[str] = set()
    for emb in embeddings:
        all_features.update(emb.features.keys())
    vocabulary = sorted(all_features)

    feature_matrix = WLFeatureMatrix(
        embeddings=embeddings,
        vocabulary=vocabulary,
        slice_labels=slice_labels,
    )

    # Cache result
    if cache_dir:
        cache.put(feature_matrix, graph_list, depth)

    return feature_matrix
