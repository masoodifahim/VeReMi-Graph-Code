"""One-epoch PyG edge-classification functionality test on released demo graphs.

Run from the repository root:
    python examples/pyg_training_smoke_test.py

This is a software smoke test, not a detection-performance benchmark.
Only load trusted .pt files: torch.load(..., weights_only=False) uses pickle.
"""
from pathlib import Path
from time import perf_counter
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("small", "medium", "large")


class EdgeGCN(nn.Module):
    def __init__(self, node_features: int, edge_features: int, hidden: int = 32):
        super().__init__()
        self.conv = GCNConv(node_features, hidden)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden + edge_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch):
        h = F.relu(self.conv(batch.x.float(), batch.edge_index))
        source, target = batch.edge_index
        combined = torch.cat((h[source], h[target], batch.edge_attr.float()), dim=1)
        return self.classifier(combined).flatten()


def main():
    torch.manual_seed(20260811)
    started = perf_counter()
    graphs = []
    for category in CATEGORIES:
        path = ROOT / "pyg" / f"{category}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing demo graph: {path}")
        # Only load the trusted dataset's official release files.
        graph = torch.load(path, map_location="cpu", weights_only=False)
        assert graph.x.ndim == 2 and graph.x.shape[1] == 26, category
        assert graph.edge_index.ndim == 2 and graph.edge_index.shape[0] == 2, category
        assert graph.edge_attr.ndim == 2 and graph.edge_attr.shape[1] == 23, category
        assert graph.edge_attr.shape[0] == graph.edge_index.shape[1], category
        assert graph.edge_y_binary.numel() == graph.edge_index.shape[1], category
        assert graph.edge_index.shape[1] > 0, category
        labels = graph.edge_y_binary.reshape(-1)
        assert torch.isfinite(graph.x).all() and torch.isfinite(graph.edge_attr).all(), category
        assert torch.isin(labels, torch.tensor([0, 1], dtype=labels.dtype)).all(), category
        graphs.append(graph)
        print(f"Loaded {category}: nodes={graph.num_nodes}, edges={graph.num_edges}")

    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    model = EdgeGCN(node_features=26, edge_features=23)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    processed_edges = 0
    batches = 0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch)
        targets = batch.edge_y_binary.reshape(-1).float()
        assert logits.shape == targets.shape
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        assert torch.isfinite(loss), "Non-finite training loss"
        loss.backward()
        optimizer.step()
        batches += 1
        processed_edges += targets.numel()
        print(f"Batch {batches}: graphs={batch.num_graphs}, edges={targets.numel()}, loss={loss.item():.6f}")

    assert batches == 2
    print(f"PASS: one epoch, {batches} batches, {processed_edges} edges, elapsed={perf_counter()-started:.2f}s")
    print(f"Versions: PyTorch={torch.__version__}; PyG={__import__('torch_geometric').__version__}")


if __name__ == "__main__":
    main()
