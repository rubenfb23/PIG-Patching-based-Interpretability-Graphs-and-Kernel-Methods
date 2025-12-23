"""Unit tests for pig.embeddings module."""

import tempfile

import numpy as np
import pytest

from pig.embeddings import (
    WLEmbedding,
    WLEmbeddingCache,
    WLEncoder,
    WLFeatureMatrix,
    compute_wl_features,
)
from pig.graph import Edge, Node, PatchInfluenceGraph
from pig.prompts import SliceLabel


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    nodes = [Node(layer=l, token=t) for l in range(2) for t in range(3)]
    edges = [
        Edge(src=0, dst=3, weight=0.8),
        Edge(src=1, dst=4, weight=0.6),
        Edge(src=2, dst=5, weight=0.4),
        Edge(src=0, dst=4, weight=0.3),
    ]
    return PatchInfluenceGraph(
        nodes=nodes,
        edges=edges,
        slice_label=SliceLabel(task="ioi", corruption="name_swap"),
        num_layers=2,
        num_tokens=3,
    )


@pytest.fixture
def sample_graph2():
    """Create a second graph with different structure."""
    nodes = [Node(layer=l, token=t) for l in range(2) for t in range(3)]
    edges = [
        Edge(src=0, dst=5, weight=0.9),
        Edge(src=1, dst=3, weight=0.5),
        Edge(src=2, dst=4, weight=0.7),
    ]
    return PatchInfluenceGraph(
        nodes=nodes,
        edges=edges,
        slice_label=SliceLabel(task="ioi", corruption="abba"),
        num_layers=2,
        num_tokens=3,
    )


class TestWLEncoder:
    """Tests for WLEncoder class."""

    def test_encode_returns_embedding(self, sample_graph):
        """Test that encode returns a WLEmbedding."""
        encoder = WLEncoder(depth=2)
        emb = encoder.encode(sample_graph)

        assert isinstance(emb, WLEmbedding)
        assert emb.depth == 2
        assert len(emb.features) > 0

    def test_deterministic(self, sample_graph):
        """Test that encoding is deterministic."""
        encoder = WLEncoder(depth=2)

        emb1 = encoder.encode(sample_graph)
        emb2 = encoder.encode(sample_graph)

        assert emb1.features == emb2.features

    def test_different_depths(self, sample_graph):
        """Test that different depths produce different features."""
        encoder1 = WLEncoder(depth=1)
        encoder2 = WLEncoder(depth=3)

        emb1 = encoder1.encode(sample_graph)
        emb2 = encoder2.encode(sample_graph)

        # Different depths should have different feature counts
        assert emb1.depth != emb2.depth
        # Deeper should have more features
        assert len(emb2.features) >= len(emb1.features)

    def test_different_graphs(self, sample_graph, sample_graph2):
        """Test that different graphs produce different features."""
        encoder = WLEncoder(depth=2)

        emb1 = encoder.encode(sample_graph)
        emb2 = encoder.encode(sample_graph2)

        # Graphs have same node structure but different edges
        # Features should differ
        assert emb1.features != emb2.features

    def test_encode_batch(self, sample_graph, sample_graph2):
        """Test batch encoding."""
        encoder = WLEncoder(depth=2)
        embeddings = encoder.encode_batch([sample_graph, sample_graph2])

        assert len(embeddings) == 2
        assert all(isinstance(e, WLEmbedding) for e in embeddings)


class TestWLEmbedding:
    """Tests for WLEmbedding class."""

    def test_to_vector(self):
        """Test conversion to dense vector."""
        features = {"f1": 3, "f2": 1, "f3": 5}
        emb = WLEmbedding(features=features, graph_id="test", depth=2)

        vocabulary = ["f1", "f2", "f3", "f4"]
        vec = emb.to_vector(vocabulary)

        assert vec.shape == (4,)
        assert vec[0] == 3  # f1
        assert vec[1] == 1  # f2
        assert vec[2] == 5  # f3
        assert vec[3] == 0  # f4 (missing)

    def test_to_dict_from_dict(self):
        """Test serialization roundtrip."""
        features = {"f1": 3, "f2": 1}
        emb = WLEmbedding(features=features, graph_id="test", depth=2)

        data = emb.to_dict()
        restored = WLEmbedding.from_dict(data)

        assert restored.features == emb.features
        assert restored.graph_id == emb.graph_id
        assert restored.depth == emb.depth


class TestWLFeatureMatrix:
    """Tests for WLFeatureMatrix class."""

    def test_to_matrix(self, sample_graph, sample_graph2):
        """Test conversion to dense matrix."""
        encoder = WLEncoder(depth=2)
        embeddings = encoder.encode_batch([sample_graph, sample_graph2])

        # Build vocabulary
        all_features = set()
        for emb in embeddings:
            all_features.update(emb.features.keys())
        vocabulary = sorted(all_features)

        fm = WLFeatureMatrix(
            embeddings=embeddings,
            vocabulary=vocabulary,
            slice_labels=[sample_graph.slice_label, sample_graph2.slice_label],
        )

        matrix = fm.to_matrix()

        assert matrix.shape == (2, len(vocabulary))
        assert matrix.dtype == np.float32

    def test_properties(self, sample_graph, sample_graph2):
        """Test matrix properties."""
        encoder = WLEncoder(depth=2)
        embeddings = encoder.encode_batch([sample_graph, sample_graph2])

        all_features = set()
        for emb in embeddings:
            all_features.update(emb.features.keys())

        fm = WLFeatureMatrix(
            embeddings=embeddings,
            vocabulary=sorted(all_features),
            slice_labels=[sample_graph.slice_label, sample_graph2.slice_label],
        )

        assert fm.num_graphs == 2
        assert fm.num_features == len(all_features)


class TestWLEmbeddingCache:
    """Tests for WLEmbeddingCache class."""

    def test_put_and_get(self, sample_graph, sample_graph2):
        """Test caching feature matrices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = WLEmbeddingCache(tmpdir)
            graphs = [sample_graph, sample_graph2]
            depth = 2

            encoder = WLEncoder(depth=depth)
            embeddings = encoder.encode_batch(graphs)

            all_features = set()
            for emb in embeddings:
                all_features.update(emb.features.keys())

            fm = WLFeatureMatrix(
                embeddings=embeddings,
                vocabulary=sorted(all_features),
                slice_labels=[g.slice_label for g in graphs],
            )

            # Initially not cached
            assert cache.get(graphs, depth) is None

            # Store
            cache.put(fm, graphs, depth)

            # Now should be retrievable
            cached = cache.get(graphs, depth)
            assert cached is not None
            assert cached.num_graphs == fm.num_graphs
            assert cached.num_features == fm.num_features


class TestComputeWLFeatures:
    """Tests for compute_wl_features function."""

    def test_basic(self, sample_graph, sample_graph2):
        """Test basic feature computation."""
        graphs = {
            sample_graph.slice_label: sample_graph,
            sample_graph2.slice_label: sample_graph2,
        }

        fm = compute_wl_features(graphs, depth=2)

        assert fm.num_graphs == 2
        assert fm.num_features > 0

        matrix = fm.to_matrix()
        assert matrix.shape[0] == 2
        assert not np.allclose(matrix[0], matrix[1])  # Should differ

    def test_with_cache(self, sample_graph, sample_graph2):
        """Test caching behavior."""
        graphs = {
            sample_graph.slice_label: sample_graph,
            sample_graph2.slice_label: sample_graph2,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # First call computes
            fm1 = compute_wl_features(graphs, depth=2, cache_dir=tmpdir)

            # Second call should use cache
            fm2 = compute_wl_features(graphs, depth=2, cache_dir=tmpdir)

            assert fm1.num_features == fm2.num_features
            np.testing.assert_array_equal(fm1.to_matrix(), fm2.to_matrix())
