"""Unit tests for pig.graph module."""

import numpy as np
import pytest

from pig.graph import Edge, GraphBuilder, Node, PatchInfluenceGraph
from pig.patching import PatchEffectDataset, PatchEffectTensor
from pig.prompts import PromptPair, SliceLabel


class TestNode:
    """Tests for Node class."""

    def test_creation(self):
        """Test node creation."""
        node = Node(layer=5, token=3)
        assert node.layer == 5
        assert node.token == 3
        assert node.node_type == "res"

    def test_ordering(self):
        """Test node ordering for direction constraint."""
        n1 = Node(layer=0, token=0)
        n2 = Node(layer=0, token=1)
        n3 = Node(layer=1, token=0)

        assert n1 < n2  # Same layer, earlier token
        assert n1 < n3  # Earlier layer
        assert n2 < n3  # Earlier layer beats later token

    def test_to_from_index(self):
        """Test index conversion."""
        node = Node(layer=3, token=5)
        num_tokens = 10

        idx = node.to_index(num_tokens)
        restored = Node.from_index(idx, num_tokens)

        assert restored.layer == node.layer
        assert restored.token == node.token


class TestEdge:
    """Tests for Edge class."""

    def test_creation(self):
        """Test edge creation."""
        edge = Edge(src=0, dst=5, weight=0.75)
        assert edge.src == 0
        assert edge.dst == 5
        assert edge.weight == 0.75

    def test_to_dict(self):
        """Test edge serialization."""
        edge = Edge(src=1, dst=2, weight=0.5)
        d = edge.to_dict()
        assert d == {"src": 1, "dst": 2, "weight": 0.5}


class TestPatchInfluenceGraph:
    """Tests for PatchInfluenceGraph class."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph for testing."""
        nodes = [Node(layer=l, token=t) for l in range(2) for t in range(3)]
        edges = [
            Edge(src=0, dst=3, weight=0.8),
            Edge(src=1, dst=4, weight=0.6),
            Edge(src=2, dst=5, weight=0.4),
        ]
        return PatchInfluenceGraph(
            nodes=nodes,
            edges=edges,
            slice_label=SliceLabel(task="ioi", corruption="name_swap"),
            num_layers=2,
            num_tokens=3,
        )

    def test_properties(self, sample_graph):
        """Test graph properties."""
        assert sample_graph.num_nodes == 6
        assert sample_graph.num_edges == 3
        assert len(sample_graph) == 6

    def test_adjacency_matrix(self, sample_graph):
        """Test adjacency matrix construction."""
        adj = sample_graph.get_adjacency_matrix()
        assert adj.shape == (6, 6)
        assert adj[0, 3] == pytest.approx(0.8)
        assert adj[1, 4] == pytest.approx(0.6)
        assert adj[0, 0] == 0.0  # No self-loops in this graph

    def test_edge_list(self, sample_graph):
        """Test edge list export."""
        edges = sample_graph.get_edge_list()
        assert len(edges) == 3
        assert (0, 3, 0.8) in edges

    def test_node_degrees(self, sample_graph):
        """Test degree computation."""
        in_deg, out_deg = sample_graph.get_node_degrees()

        assert len(in_deg) == 6
        assert len(out_deg) == 6

        # Sources have out_degree 1
        assert out_deg[0] == 1
        assert out_deg[1] == 1
        assert out_deg[2] == 1

        # Destinations have in_degree 1
        assert in_deg[3] == 1
        assert in_deg[4] == 1
        assert in_deg[5] == 1

    def test_statistics(self, sample_graph):
        """Test statistics computation."""
        stats = sample_graph.compute_statistics()

        assert stats["num_nodes"] == 6
        assert stats["num_edges"] == 3
        assert stats["mean_weight"] == pytest.approx(0.6)

    def test_to_dict(self, sample_graph):
        """Test serialization."""
        d = sample_graph.to_dict()

        assert len(d["nodes"]) == 6
        assert len(d["edges"]) == 3
        assert d["slice"]["task"] == "ioi"


class TestGraphBuilder:
    """Tests for GraphBuilder class."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for graph building."""
        dataset = PatchEffectDataset()
        slice_label = SliceLabel(task="ioi", corruption="name_swap")

        # Create 10 examples with correlated effects
        np.random.seed(42)
        for i in range(10):
            # Create effects with some structure
            effects = np.random.randn(4, 5).astype(np.float32)
            # Add correlation between certain positions
            effects[0, :] += 0.5 * effects[1, :]
            effects[2, :] += 0.5 * effects[3, :]

            pair = PromptPair(
                x_cln=f"Clean {i}",
                x_crp=f"Corrupt {i}",
                y_star="target",
                slice_label=slice_label,
                meta={},
            )
            tensor = PatchEffectTensor(
                effects=effects,
                prompt_pair=pair,
                base_score=-80.0,
                clean_score=-70.0,
            )
            dataset.add(tensor)

        return dataset

    def test_build_from_slice(self, sample_dataset):
        """Test building a graph from a slice."""
        builder = GraphBuilder(k=3, enforce_direction=True)
        slice_label = SliceLabel(task="ioi", corruption="name_swap")

        graph = builder.build_from_slice(sample_dataset, slice_label)

        assert isinstance(graph, PatchInfluenceGraph)
        assert graph.num_nodes == 4 * 5  # 4 layers * 5 tokens
        assert graph.slice_label == slice_label

    def test_direction_constraint(self, sample_dataset):
        """Test that direction constraint is enforced."""
        builder = GraphBuilder(k=10, enforce_direction=True)
        slice_label = SliceLabel(task="ioi", corruption="name_swap")

        graph = builder.build_from_slice(sample_dataset, slice_label)

        # All edges should go from earlier to later nodes
        for edge in graph.edges:
            src_node = graph.nodes[edge.src]
            dst_node = graph.nodes[edge.dst]
            assert src_node < dst_node, f"Edge {edge} violates direction"

    def test_no_direction_constraint(self, sample_dataset):
        """Test building without direction constraint."""
        builder = GraphBuilder(k=10, enforce_direction=False)
        slice_label = SliceLabel(task="ioi", corruption="name_swap")

        graph = builder.build_from_slice(sample_dataset, slice_label)

        # Should have more edges without constraint
        assert graph.num_edges > 0

    def test_topk_sparsification(self, sample_dataset):
        """Test that top-k limits edges per node."""
        k = 3
        builder = GraphBuilder(k=k, enforce_direction=False)
        slice_label = SliceLabel(task="ioi", corruption="name_swap")

        graph = builder.build_from_slice(sample_dataset, slice_label)

        # Count outgoing edges per node
        out_edges = {}
        for edge in graph.edges:
            out_edges[edge.src] = out_edges.get(edge.src, 0) + 1

        # No node should have more than k outgoing edges
        for node, count in out_edges.items():
            assert count <= k, f"Node {node} has {count} > {k} outgoing edges"

    def test_build_all(self, sample_dataset):
        """Test building graphs for all slices."""
        builder = GraphBuilder(k=3)

        graphs = builder.build_all(sample_dataset)

        assert len(graphs) == 1
        slice_label = SliceLabel(task="ioi", corruption="name_swap")
        assert slice_label in graphs
