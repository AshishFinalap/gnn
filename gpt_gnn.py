# gnn.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from typing import Optional, Tuple, List
import numpy as np
import random

# -------------------------
# Model
# -------------------------
class GraphSAGENodePredictor(nn.Module):
    """
    GraphSAGE-based node-level predictor.
    Produces a scalar priority score per node in [0, 1].

    Args:
        in_channels: int, number of node features
        hidden_channels: int, hidden dimension for SAGEConv layers
        num_layers: int, number of SAGEConv layers (>=1)
        dropout: float, dropout probability
        aggr: str, aggregation method for SAGEConv ('mean', 'max', 'sum')
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggr: str = 'mean'
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.num_layers = num_layers
        self.dropout = dropout

        # Build SAGEConv layers
        self.convs = nn.ModuleList()
        if num_layers == 1:
            # single layer maps in_channels -> hidden (then directly to output)
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        else:
            # first layer
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            # middle layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            # last conv
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

        # BatchNorm for hidden representation
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])

        # Final MLP head to scalar output
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2 if hidden_channels // 2 > 0 else 1),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2 if hidden_channels // 2 > 0 else 1, 1)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: torch_geometric.data.Data with attributes:
                  - x: node features [num_nodes, num_features]
                  - edge_index: [2, num_edges]
        Returns:
            node_scores: tensor shape [num_nodes], values in [0,1]
        """
        x, edge_index = data.x, data.edge_index

        # Message-passing through SAGEConv layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # BatchNorm expects shape [N, features]; if single node, guard it
            if x.size(0) > 1:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # MLP head to scalar
        out = self.mlp(x)  # shape [num_nodes, 1]
        out = out.squeeze(dim=-1)  # [num_nodes]
        scores = torch.sigmoid(out)  # map to [0,1] -> priority score
        return scores


# -------------------------
# Training / evaluation helpers
# -------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    train_idx: Optional[torch.Tensor] = None,
    val_idx: Optional[torch.Tensor] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[nn.Module, dict]:
    """
    Train node-level regression (priority) model.
    labels: tensor shape [num_nodes], float in [0,1]
    train_idx/val_idx: optional LongTensor indices of training / validation nodes.
                      If None, all nodes used for training (not recommended).
    Returns:
        model: trained model
        history: dict with train_loss, val_loss lists
    """
    model = model.to(device)
    data = data.to(device)
    labels = labels.to(device)

    if train_idx is None:
        train_idx = torch.arange(data.num_nodes, device=device)
    if val_idx is None:
        # small default split: last 10% as val if possible
        n = data.num_nodes
        k = max(1, n // 10)
        perm = torch.randperm(n, device=device)
        val_idx = perm[:k]
        train_idx = perm[k:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()  # regression of priority score in [0,1]

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out_scores = model(data)  # [num_nodes]
        loss = loss_fn(out_scores[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            out_val = model(data)
            val_loss = loss_fn(out_val[val_idx], labels[val_idx]).item()

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch <= 5):
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    return model, history


def evaluate_model(
    model: nn.Module,
    data: Data,
    labels: torch.Tensor,
    test_idx: Optional[torch.Tensor] = None,
    device: str = 'cpu'
) -> dict:
    """
    Evaluate model on test indices. Returns MSE and MAE and optional ranking metrics.
    """
    model = model.to(device)
    data = data.to(device)
    labels = labels.to(device)

    if test_idx is None:
        test_idx = torch.arange(data.num_nodes, device=device)

    model.eval()
    with torch.no_grad():
        preds = model(data)
        preds_t = preds[test_idx]
        y_t = labels[test_idx]
        mse = F.mse_loss(preds_t, y_t).item()
        mae = F.l1_loss(preds_t, y_t).item()

    # Optionally compute top-k recall if labels are binary/high-priority; skip here
    return {'mse': mse, 'mae': mae}


def predict_node_scores(model: nn.Module, data: Data, device: str = 'cpu') -> np.ndarray:
    """
    Run model in eval mode and return numpy array of node priority scores [num_nodes].
    """
    model = model.to(device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        scores = model(data).cpu().numpy()
    return scores


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str, device: str = 'cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model


# -------------------------
# Utility: build a Pyg Data from numpy (helper)
# -------------------------
def build_data_from_numpy(
    features: np.ndarray,
    adj_matrix: np.ndarray,
    device: str = 'cpu'
) -> Data:
    """
    Create torch_geometric.data.Data from numpy features and adjacency matrix.
    features: [num_nodes, num_features]
    adj_matrix: [num_nodes, num_nodes] binary (0/1)
    """
    x = torch.tensor(features, dtype=torch.float)
    row, col = np.where(adj_matrix == 1)
    edge_index = torch.tensor(np.vstack((row, col)), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    return data


# -------------------------
# Example / quick test (toy graph)
# -------------------------
if __name__ == "__main__":
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Toy features (4 nodes) -> [distance, temperature, movement, sound]
    features = np.array([
        [0.8, 0.7, 0.4, 0.9],  # Node 0
        [0.0, 0.2, 0.0, 0.0],  # Node 1 (safe)
        [0.5, 0.5, 0.3, 0.2],  # Node 2
        [0.7, 0.9, 0.6, 1.0],  # Node 3 (high priority)
    ], dtype=np.float32)

    # Toy adjacency (undirected)
    adj = np.array([
        [0,1,1,0],
        [1,0,0,1],
        [1,0,0,1],
        [0,1,1,0]
    ], dtype=np.int32)

    data = build_data_from_numpy(features, adj, device=device)

    # Toy "labels" (priority) - if you have ground truth, put here, else synthetic
    # Example: Node 3 highest, Node 0 high, others low
    labels = torch.tensor([0.78, 0.05, 0.40, 0.95], dtype=torch.float)

    # Train / val splits
    n = data.num_nodes
    idx = torch.randperm(n)
    train_idx = idx[:max(1, int(0.75*n))]
    val_idx = idx[max(1, int(0.75*n)):]

    model = GraphSAGENodePredictor(in_channels=features.shape[1], hidden_channels=32, num_layers=2, dropout=0.3)
    model, history = train_model(model, data, labels, train_idx=train_idx, val_idx=val_idx, lr=1e-3, epochs=200, device=device, verbose=True)

    print("Evaluation:", evaluate_model(model, data, labels, test_idx=None, device=device))
    preds = predict_node_scores(model, data, device=device)
    print("Predicted scores:", preds)
    # Node id mapping: if you used build_pyg_graph, you can attach data.node_ids = nodes
