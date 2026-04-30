"""Small PyTorch GNN baseline for PIG graphs.

The implementation intentionally avoids external graph-learning dependencies.
It is meant as an auxiliary baseline, not as the main causal method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional as F

from pig.graph import PatchInfluenceGraph
from pig.prompts import SliceLabel

NODE_TYPE_FEATURES = ("att", "mlp", "res")


@dataclass
class GNNTrainingConfig:
    """Training settings for the lightweight GNN classifier."""

    hidden_dim: int = 32
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-2
    weight_decay: float = 1e-3
    epochs: int = 150
    random_state: int = 42
    device: str = "cpu"


@dataclass
class GNNCVResult:
    """Cross-validation result for the GNN baseline."""

    accuracy_mean: float
    accuracy_std: float
    fold_accuracies: list[float]
    cv_folds: int
    num_graphs: int

    def to_dict(self) -> dict:
        return {
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "fold_accuracies": self.fold_accuracies,
            "cv_folds": self.cv_folds,
            "num_graphs": self.num_graphs,
        }


@dataclass
class GNNEncoderSVMResult:
    """Example-disjoint result for a supervised GNN encoder followed by SVM."""

    accuracy: float
    num_train: int
    num_test: int
    embedding_dim: int
    svm_kernel: str

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "num_train": self.num_train,
            "num_test": self.num_test,
            "embedding_dim": self.embedding_dim,
            "svm_kernel": self.svm_kernel,
        }


class SimpleMessagePassingGNN(nn.Module):
    """Directed weighted message-passing graph classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.self_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.msg_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        graph_repr = self.encode(node_features, adjacency, mask)
        return self.classifier(graph_repr)

    def encode(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the pooled graph representation before the classifier."""
        h = F.relu(self.input_proj(node_features))
        h = h * mask.unsqueeze(-1)

        for self_layer, msg_layer in zip(self.self_layers, self.msg_layers):
            messages = torch.bmm(adjacency, h)
            h = F.relu(self_layer(h) + msg_layer(messages))
            h = self.dropout(h) * mask.unsqueeze(-1)

        counts = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_mean = h.sum(dim=1) / counts
        pooled_max = h.masked_fill(~mask.unsqueeze(-1).bool(), -1e9).max(dim=1).values
        pooled_max = torch.where(
            torch.isfinite(pooled_max), pooled_max, torch.zeros_like(pooled_max)
        )
        graph_repr = torch.cat([pooled_mean, pooled_max], dim=-1)
        return graph_repr


def cross_validate_gnn_baseline(
    graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    *,
    config: Optional[GNNTrainingConfig] = None,
    cv: int = 5,
) -> GNNCVResult:
    """Train/evaluate a lightweight GNN with stratified CV."""
    if config is None:
        config = GNNTrainingConfig()
    if not graphs_with_labels:
        raise ValueError("graphs_with_labels must not be empty")

    graphs = [graph for graph, _ in graphs_with_labels]
    labels = [str(label) for _, label in graphs_with_labels]
    classes = sorted(set(labels))
    class_to_id = {label: idx for idx, label in enumerate(classes)}
    y = np.array([class_to_id[label] for label in labels], dtype=np.int64)

    _, class_counts = np.unique(y, return_counts=True)
    max_splits = int(min(np.min(class_counts), len(y)))
    if max_splits < 2:
        raise ValueError("Need at least two samples per class for GNN CV")
    cv_splitter = StratifiedKFold(
        n_splits=min(cv, max_splits),
        shuffle=True,
        random_state=config.random_state,
    )

    fold_accuracies = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        cv_splitter.split(np.zeros(len(y)), y)
    ):
        fold_seed = config.random_state + fold_idx
        _seed_torch(fold_seed)
        fold_config = GNNTrainingConfig(
            **{**config.__dict__, "random_state": fold_seed}
        )
        accuracy = _train_eval_fold(
            [graphs[idx] for idx in train_idx],
            y[train_idx],
            [graphs[idx] for idx in test_idx],
            y[test_idx],
            num_classes=len(classes),
            config=fold_config,
        )
        fold_accuracies.append(float(accuracy))

    return GNNCVResult(
        accuracy_mean=float(np.mean(fold_accuracies)),
        accuracy_std=float(np.std(fold_accuracies)),
        fold_accuracies=fold_accuracies,
        cv_folds=int(cv_splitter.get_n_splits()),
        num_graphs=len(graphs_with_labels),
    )


def fit_gnn_encoder_svm(
    train_graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    test_graphs_with_labels: list[tuple[PatchInfluenceGraph, SliceLabel]],
    *,
    config: Optional[GNNTrainingConfig] = None,
    svm_kernel: str = "linear",
) -> GNNEncoderSVMResult:
    """Train a supervised GNN encoder on train graphs and classify embeddings with SVM."""
    if config is None:
        config = GNNTrainingConfig()
    if not train_graphs_with_labels or not test_graphs_with_labels:
        raise ValueError("train and test graph lists must not be empty")

    train_graphs = [graph for graph, _ in train_graphs_with_labels]
    test_graphs = [graph for graph, _ in test_graphs_with_labels]
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(
        [str(label) for _, label in train_graphs_with_labels]
    )
    y_test = label_encoder.transform(
        [str(label) for _, label in test_graphs_with_labels]
    )

    train_embeddings, test_embeddings = _train_encoder_embeddings(
        train_graphs,
        y_train,
        test_graphs,
        num_classes=len(label_encoder.classes_),
        config=config,
    )
    estimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel=svm_kernel,
                    C=1.0,
                    gamma="scale",
                    random_state=config.random_state,
                    probability=False,
                ),
            ),
        ]
    )
    estimator.fit(train_embeddings, y_train)
    predictions = estimator.predict(test_embeddings)
    return GNNEncoderSVMResult(
        accuracy=float(accuracy_score(y_test, predictions)),
        num_train=int(train_embeddings.shape[0]),
        num_test=int(test_embeddings.shape[0]),
        embedding_dim=int(train_embeddings.shape[1]),
        svm_kernel=svm_kernel,
    )


def _train_encoder_embeddings(
    train_graphs: list[PatchInfluenceGraph],
    train_labels: np.ndarray,
    test_graphs: list[PatchInfluenceGraph],
    *,
    num_classes: int,
    config: GNNTrainingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(config.device)
    _seed_torch(config.random_state)
    train_batch = _graphs_to_batch(train_graphs, device=device)
    test_batch = _graphs_to_batch(test_graphs, device=device)
    y_train = torch.as_tensor(train_labels, dtype=torch.long, device=device)

    model = SimpleMessagePassingGNN(
        input_dim=train_batch[0].shape[-1],
        hidden_dim=config.hidden_dim,
        num_classes=num_classes,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(*train_batch)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_embeddings = model.encode(*train_batch).detach().cpu().numpy()
        test_embeddings = model.encode(*test_batch).detach().cpu().numpy()
    return train_embeddings, test_embeddings


def _train_eval_fold(
    train_graphs: list[PatchInfluenceGraph],
    train_labels: np.ndarray,
    test_graphs: list[PatchInfluenceGraph],
    test_labels: np.ndarray,
    *,
    num_classes: int,
    config: GNNTrainingConfig,
) -> float:
    device = torch.device(config.device)
    train_batch = _graphs_to_batch(train_graphs, device=device)
    test_batch = _graphs_to_batch(test_graphs, device=device)
    y_train = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    y_test = torch.as_tensor(test_labels, dtype=torch.long, device=device)

    model = SimpleMessagePassingGNN(
        input_dim=train_batch[0].shape[-1],
        hidden_dim=config.hidden_dim,
        num_classes=num_classes,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(*train_batch)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(*test_batch)
        predictions = logits.argmax(dim=-1)
        return float((predictions == y_test).float().mean().item())


def _graphs_to_batch(
    graphs: list[PatchInfluenceGraph],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_nodes = max(graph.num_nodes for graph in graphs)
    feature_dim = _node_feature_dim()
    node_features = torch.zeros(
        (len(graphs), max_nodes, feature_dim),
        dtype=torch.float32,
        device=device,
    )
    adjacency = torch.zeros(
        (len(graphs), max_nodes, max_nodes),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros((len(graphs), max_nodes), dtype=torch.float32, device=device)

    for batch_idx, graph in enumerate(graphs):
        n = graph.num_nodes
        mask[batch_idx, :n] = 1.0
        for node_idx, node in enumerate(graph.nodes):
            node_features[batch_idx, node_idx] = torch.as_tensor(
                _node_features(graph, node_idx),
                dtype=torch.float32,
                device=device,
            )
        for edge in graph.edges:
            adjacency[batch_idx, edge.dst, edge.src] = float(edge.weight)

    denom = adjacency.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
    adjacency = adjacency / denom
    return node_features, adjacency, mask


def _node_features(graph: PatchInfluenceGraph, node_idx: int) -> np.ndarray:
    node = graph.nodes[node_idx]
    layer_den = max(graph.num_layers - 1, 1)
    token_den = max(graph.num_tokens - 1, 1)
    node_type = np.zeros(len(NODE_TYPE_FEATURES), dtype=np.float32)
    if node.node_type in NODE_TYPE_FEATURES:
        node_type[NODE_TYPE_FEATURES.index(node.node_type)] = 1.0
    head_value = -1.0 if node.head is None else float(node.head)
    return np.array(
        [
            node.layer / layer_den,
            node.token / token_den,
            head_value,
            1.0,
            *node_type.tolist(),
        ],
        dtype=np.float32,
    )


def _node_feature_dim() -> int:
    return 4 + len(NODE_TYPE_FEATURES)


def _seed_torch(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
