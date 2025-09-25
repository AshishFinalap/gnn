"""
gnn.py — GraphSAGE-based node priority predictor for earthquake rescue routing.

Dependencies:
- Python >= 3.8
- torch >= 1.12
- torch_geometric (compatible with torch version)
- numpy
- networkx

Optional:
- pandas (for CSV loading elsewhere)

Author: SIH Hackathon Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import random
from typing import Optional, Tuple, Dict, Union
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx

# -------------------------------
# 1. GraphSAGE Model Definition
# -------------------------------

class GraphSAGENodePredictor(nn.Module):
    """
    GraphSAGE-based node-level regression model to predict rescue priority scores in [0,1].
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2,
                 dropout: float = 0.5, aggr: str = 'mean'):
        super(GraphSAGENodePredictor, self).__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        # Final MLP head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, data: Union[Data, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass. Accepts either a Data object or (x, edge_index) tuple.
        Returns: Tensor of shape [num_nodes] with scores in [0,1].
        """
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        else:
            x, edge_index = data

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.head(x).squeeze()
        return out  # shape: [num_nodes]

# -------------------------------
# 2. Training & Evaluation Utils
# -------------------------------

def seed_everything(seed: int = 42):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(model, data, labels, train_idx=None, val_idx=None,
                lr=1e-3, weight_decay=1e-5, epochs=200, device='cpu', verbose=True):
    """
    Trains the model using MSE loss for node-level regression.
    Returns: (trained_model, history_dict)
    """
    model = model.to(device)
    data = data.to(device)
    labels = labels.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    num_nodes = data.num_nodes
    if train_idx is None or val_idx is None:
        indices = torch.randperm(num_nodes)
        split = int(0.8 * num_nodes)
        train_idx = indices[:split]
        val_idx = indices[split:]

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = criterion(val_out[val_idx], labels[val_idx])

        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    return model, history

def evaluate_model(model, data, labels, test_idx=None, device='cpu'):
    """
    Evaluates model performance. Returns MSE and MAE.
    """
    model.eval()
    model = model.to(device)
    data = data.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        preds = model(data)

    if test_idx is not None:
        preds = preds[test_idx]
        labels = labels[test_idx]

    mse = F.mse_loss(preds, labels).item()
    mae = F.l1_loss(preds, labels).item()

    return {'mse': mse, 'mae': mae}

def predict_node_scores(model, data, device='cpu') -> np.ndarray:
    """
    Predicts node scores in [0,1]. Returns numpy array of shape [num_nodes].
    """
    model.eval()
    model = model.to(device)
    data = data.to(device)

    with torch.no_grad():
        scores = model(data).cpu().numpy()

    return scores

def save_model(model, path: str):
    """
    Saves model to disk.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path: str, device='cpu'):
    """
    Loads model weights from disk.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

# -------------------------------
# 3. Data Utilities
# -------------------------------

def build_data_from_numpy(features: np.ndarray, adj_matrix: np.ndarray) -> Data:
    """
    Converts numpy features and adjacency matrix to PyG Data object.
    """
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array of shape [num_nodes, num_features].")
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    if not np.allclose(adj_matrix, adj_matrix.T):
        raise ValueError("Adjacency matrix must be symmetric (undirected graph).")

    # Handle NaNs
    if np.isnan(features).any():
        col_means = np.nanmean(features, axis=0)
        inds = np.where(np.isnan(features))
        features[inds] = np.take(col_means, inds[1])

    edge_index = np.array(np.nonzero(adj_matrix))
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x = torch.tensor(features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)

    return data

# -------------------------------
# 4. Dijkstra Integration Helper
# -------------------------------

def node_scores_to_edge_weights(data: Data, node_scores: np.ndarray,
                                method: str = 'avg', alpha: float = 1.0, beta: float = 1.0) -> nx.Graph:
    """
    Converts node scores to edge weights for Dijkstra.
    Returns a NetworkX graph with 'weight' on each edge.
    """
    G = nx.Graph()
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        G.add_node(i)

    edge_index = data.edge_index.cpu().numpy()
    for u, v in edge_index.T:
        score_u = node_scores[u]
        score_v = node_scores[v]
        priority_penalty = ((1 - score_u) + (1 - score_v)) / 2

        # Optional: use distance feature if available
        if data.x.shape[1] >= 1:
            distance_u = data.x[u][0].item()
            distance_v = data.x[v][0].item()
            edge_distance = (distance_u + distance_v) / 2
        else:
            edge_distance = 1.0

        weight = alpha * edge_distance + beta * priority_penalty
        G.add_edge(u, v, weight=weight)

    return G

# -------------------------------
# 5. Toy Example
# -------------------------------

if __name__ == "__main__":
    print("Running toy example...")

    seed_everything()

    # Toy features: [distance, temperature, movement, sound]
    features = np.array([
        [0.8, 0.7, 0.4, 0.9],  # Node 0: collapsed building
        [0.0, 0.2, 0.0, 0.0],  # Node 1: safe ground
        [0.5, 0.5, 0.3, 0.2],  # Node 2: street
        [0.7, 0.9, 0.6, 1.0],  # Node 3: school
    ], dtype=np.float32)

    # Symmetric adjacency matrix
