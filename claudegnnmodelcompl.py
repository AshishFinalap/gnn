"""
gnn.py - Graph Neural Network for Earthquake Rescue Priority Prediction
========================================================================
A complete implementation of a GraphSAGE-based node priority predictor for 
drone-based earthquake rescue operations.

Dependencies:
- Python 3.8+
- torch >= 1.12
- torch_geometric (compatible with torch version)
- numpy
- networkx
- Optional: pandas (for CSV loading in integration)

Author: SIH Hackathon Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
import numpy as np
import networkx as nx
import random
import warnings
from typing import Optional, Tuple, Dict, List, Union
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GraphSAGENodePredictor(nn.Module):
    """
    GraphSAGE-based model for node-level priority prediction.
    
    Produces a scalar priority score in [0, 1] for each node in the graph,
    indicating rescue urgency based on sensor features.
    
    Args:
        in_channels: Number of input features per node (4 for [distance, temperature, movement, sound])
        hidden_channels: Hidden layer dimension (default: 64)
        num_layers: Number of GraphSAGE convolution layers (default: 2)
        dropout: Dropout probability (default: 0.5)
        aggr: Aggregation method for SAGEConv ('mean', 'max', 'add') (default: 'mean')
    """
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 aggr: str = 'mean'):
        super(GraphSAGENodePredictor, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GraphSAGE convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            
            # Add normalization after each conv layer
            self.norms.append(nn.LayerNorm(hidden_channels))
        
        # MLP head for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)  # Single scalar output
        )
        
    def forward(self, data: Union[Data, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            data: Either a torch_geometric.data.Data object or tuple (x, edge_index)
                  where x is node features [num_nodes, in_channels] and 
                  edge_index is COO format edges [2, num_edges]
        
        Returns:
            torch.Tensor: Priority scores for each node, shape [num_nodes], values in [0, 1]
        """
        # Handle both Data object and tuple input
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        else:
            x, edge_index = data
        
        # Apply GraphSAGE convolutions with normalization and activation
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            
            # Apply dropout except on the last layer
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply MLP head to get scalar predictions
        x = self.mlp(x)  # Shape: [num_nodes, 1]
        x = torch.sigmoid(x)  # Constrain to [0, 1]
        x = x.squeeze(-1)  # Shape: [num_nodes]
        
        return x


def build_data_from_numpy(features: np.ndarray, 
                         adj_matrix: np.ndarray,
                         node_ids: Optional[List] = None) -> Data:
    """
    Convert numpy arrays to PyTorch Geometric Data object.
    
    Args:
        features: Node features array, shape [num_nodes, num_features]
        adj_matrix: Adjacency matrix, shape [num_nodes, num_nodes]
        node_ids: Optional list of node identifiers (e.g., GPS IDs)
    
    Returns:
        torch_geometric.data.Data object with x and edge_index
    
    Raises:
        ValueError: If shapes are incompatible or adjacency is not symmetric
    """
    # Defensive checks
    if features.ndim != 2:
        raise ValueError(f"Features must be 2D array, got shape {features.shape}")
    
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj_matrix.shape}")
    
    if features.shape[0] != adj_matrix.shape[0]:
        raise ValueError(f"Number of nodes mismatch: features {features.shape[0]} vs adjacency {adj_matrix.shape[0]}")
    
    # Check symmetry (for undirected graph)
    if not np.allclose(adj_matrix, adj_matrix.T):
        raise ValueError("Adjacency matrix must be symmetric for undirected graph")
    
    # Handle missing values
    if np.any(np.isnan(features)):
        warnings.warn("NaN values detected in features. Filling with column means.")
        col_means = np.nanmean(features, axis=0)
        for i in range(features.shape[1]):
            features[np.isnan(features[:, i]), i] = col_means[i]
    
    # Convert to PyTorch tensors
    x = torch.tensor(features, dtype=torch.float32)
    
    # Convert adjacency matrix to edge_index (COO format)
    edges = np.where(adj_matrix > 0)
    edge_index = torch.tensor(np.vstack([edges[0], edges[1]]), dtype=torch.long)
    
    # Create Data object
    data = Data(x=x, edge_index=edge_index)
    
    # Add node IDs if provided
    if node_ids is not None:
        data.node_ids = node_ids
    
    return data


def train_model(model: GraphSAGENodePredictor,
               data: Data,
               labels: torch.Tensor,
               train_idx: Optional[torch.Tensor] = None,
               val_idx: Optional[torch.Tensor] = None,
               lr: float = 1e-3,
               weight_decay: float = 1e-5,
               epochs: int = 200,
               device: str = 'cpu',
               verbose: bool = True) -> Tuple[GraphSAGENodePredictor, Dict[str, List[float]]]:
    """
    Train the GraphSAGE model for node-level regression.
    
    Args:
        model: GraphSAGE model instance
        data: PyTorch Geometric Data object
        labels: Target priority scores for each node, shape [num_nodes], values in [0, 1]
        train_idx: Indices of training nodes (if None, uses 60% of nodes)
        val_idx: Indices of validation nodes (if None, uses 20% of nodes)
        lr: Learning rate
        weight_decay: L2 regularization weight
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        verbose: Whether to print training progress
    
    Returns:
        Tuple of (trained_model, history_dict) where history_dict contains
        'train_loss' and 'val_loss' lists
    """
    # Move to device
    model = model.to(device)
    data = data.to(device)
    labels = labels.to(device)
    
    num_nodes = data.x.shape[0]
    
    # Default train/val/test split if not provided
    if train_idx is None or val_idx is None:
        perm = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        if train_idx is None:
            train_idx = perm[:train_size]
        if val_idx is None:
            val_idx = perm[train_size:train_size + val_size]
    
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    
    model.train()
    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        out = model(data)
        train_loss = criterion(out[train_idx], labels[train_idx])
        train_loss.backward()
        optimizer.step()
        
        # Validation step
        model.eval()
        with torch.no_grad():
            out = model(data)
            val_loss = criterion(out[val_idx], labels[val_idx])
        model.train()
        
        history['train_loss'].append(train_loss.item())
        history['val_loss'].append(val_loss.item())
        
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    model.eval()
    return model, history


def evaluate_model(model: GraphSAGENodePredictor,
                  data: Data,
                  labels: torch.Tensor,
                  test_idx: Optional[torch.Tensor] = None,
                  device: str = 'cpu') -> Dict[str, float]:
    """
    Evaluate the model performance.
    
    Args:
        model: Trained GraphSAGE model
        data: PyTorch Geometric Data object
        labels: Ground truth priority scores
        test_idx: Indices of test nodes (if None, uses all nodes)
        device: Device to evaluate on
    
    Returns:
        Dictionary containing evaluation metrics (mse, mae, top_k_recall)
    """
    model = model.to(device)
    data = data.to(device)
    labels = labels.to(device)
    
    if test_idx is None:
        test_idx = torch.arange(data.x.shape[0])
    test_idx = test_idx.to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(data)
        test_predictions = predictions[test_idx]
        test_labels = labels[test_idx]
        
        mse = F.mse_loss(test_predictions, test_labels).item()
        mae = F.l1_loss(test_predictions, test_labels).item()
        
        # Top-k recall for highest priority nodes
        k = min(5, len(test_idx))
        _, pred_top_k = torch.topk(test_predictions, k)
        _, true_top_k = torch.topk(test_labels, k)
        
        recall = len(set(pred_top_k.cpu().numpy()) & set(true_top_k.cpu().numpy())) / k
    
    return {
        'mse': mse,
        'mae': mae,
        'top_5_recall': recall
    }


def predict_node_scores(model: GraphSAGENodePredictor,
                       data: Data,
                       device: str = 'cpu') -> np.ndarray:
    """
    Get priority predictions for all nodes.
    
    Args:
        model: Trained GraphSAGE model
        data: PyTorch Geometric Data object
        device: Device to run inference on
    
    Returns:
        Numpy array of shape [num_nodes] with priority scores in [0, 1]
    """
    model = model.to(device)
    data = data.to(device)
    
    model.eval()
    with torch.no_grad():
        scores = model(data)
    
    return scores.cpu().numpy()


def save_model(model: GraphSAGENodePredictor, path: str) -> None:
    """
    Save model state dict to file.
    
    Args:
        model: Model to save
        path: File path to save to
    """
    torch.save({
        'state_dict': model.state_dict(),
        'in_channels': model.in_channels,
        'hidden_channels': model.hidden_channels,
        'num_layers': model.num_layers,
        'dropout': model.dropout
    }, path)
    print(f"Model saved to {path}")


def load_model(model: GraphSAGENodePredictor, path: str, device: str = 'cpu') -> GraphSAGENodePredictor:
    """
    Load model state dict from file.
    
    Args:
        model: Model instance to load weights into
        path: File path to load from
        device: Device to load model to
    
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print(f"Model loaded from {path}")
    return model


def node_scores_to_edge_weights(data: Data,
                               node_scores: np.ndarray,
                               method: str = 'avg',
                               alpha: float = 1.0,
                               beta: float = 1.0,
                               edge_distances: Optional[np.ndarray] = None) -> nx.Graph:
    """
    Convert node priority scores to edge weights for Dijkstra pathfinding.
    
    Edge weight combines distance and inverse priority:
    weight = alpha * distance + beta * (1 - avg_priority)
    
    Lower priority nodes have higher traversal cost.
    
    Args:
        data: PyTorch Geometric Data object with edge_index
        node_scores: Priority scores for each node [0, 1]
        method: How to combine node scores ('avg', 'min', 'max')
        alpha: Weight for distance component
        beta: Weight for priority component
        edge_distances: Optional edge distances (if None, uses 1.0)
    
    Returns:
        NetworkX graph with weighted edges ready for Dijkstra
    """
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with their priority scores
    for i in range(len(node_scores)):
        G.add_node(i, priority=node_scores[i])
    
    # Process edges
    edge_index = data.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    
    # Use provided distances or default to 1.0
    if edge_distances is None:
        edge_distances = np.ones(num_edges)
    
    # Add weighted edges
    edges_seen = set()
    for i in range(num_edges):
        u, v = edge_index[0, i], edge_index[1, i]
        
        # Skip duplicate edges (undirected graph)
        if (min(u, v), max(u, v)) in edges_seen:
            continue
        edges_seen.add((min(u, v), max(u, v)))
        
        # Combine node priorities based on method
        if method == 'avg':
            combined_priority = (node_scores[u] + node_scores[v]) / 2
        elif method == 'min':
            combined_priority = min(node_scores[u], node_scores[v])
        elif method == 'max':
            combined_priority = max(node_scores[u], node_scores[v])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate edge weight (lower priority = higher cost)
        # High priority (close to 1) -> low weight (easy to traverse)
        # Low priority (close to 0) -> high weight (hard to traverse)
        priority_cost = 1.0 - combined_priority
        distance_cost = edge_distances[i] if i < len(edge_distances) else 1.0
        
        weight = alpha * distance_cost + beta * priority_cost
        
        G.add_edge(u, v, weight=weight, distance=distance_cost, priority_cost=priority_cost)
    
    return G


# Optional: Classification variant for priority bins
class GraphSAGENodeClassifier(GraphSAGENodePredictor):
    """
    Variant for classification into priority bins (low/medium/high).
    Inherits from GraphSAGENodePredictor but changes the output head.
    """
    
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 aggr: str = 'mean',
                 num_classes: int = 3):
        super().__init__(in_channels, hidden_channels, num_layers, dropout, aggr)
        
        # Replace MLP head for classification
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
        self.num_classes = num_classes
    
    def forward(self, data: Union[Data, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Returns logits for each class, shape [num_nodes, num_classes]"""
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        else:
            x, edge_index = data
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.mlp(x)  # [num_nodes, num_classes]
        return x  # Return logits (use CrossEntropyLoss which applies softmax)


if __name__ == "__main__":
    """
    Toy example demonstrating the complete pipeline.
    """
    print("=" * 70)
    print("GNN Earthquake Rescue Priority Prediction - Toy Example")
    print("=" * 70)
    
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Define toy graph with 4 nodes
    # Feature order: [distance, temperature, movement, sound]
    features = np.array([
        [0.8, 0.7, 0.4, 0.9],  # Node 0: collapsed building
        [0.0, 0.2, 0.0, 0.0],  # Node 1: safe ground
        [0.5, 0.5, 0.3, 0.2],  # Node 2: street
        [0.7, 0.9, 0.6, 1.0],  # Node 3: school (highest priority)
    ], dtype=np.float32)
    
    # Adjacency matrix (symmetric, undirected graph)
    # Connections: 0-1, 0-2, 1-2, 2-3
    adj = np.array([
        [0, 1, 1, 0],  # Node 0 connects to 1, 2
        [1, 0, 1, 0],  # Node 1 connects to 0, 2
        [1, 1, 0, 1],  # Node 2 connects to 0, 1, 3
        [0, 0, 1, 0],  # Node 3 connects to 2
    ], dtype=np.int32)
    
    # Ground truth priority labels
    labels = torch.tensor([0.78, 0.05, 0.40, 0.95], dtype=torch.float32)
    
    # Node IDs for mapping
    node_ids = ['Building_A', 'SafeZone', 'Street_B', 'School_C']
    
    print("\nNode Information:")
    print("-" * 40)
    for i, (node_id, label) in enumerate(zip(node_ids, labels)):
        print(f"Node {i} ({node_id}): Priority = {label:.2f}")
        print(f"  Features: distance={features[i,0]:.2f}, temp={features[i,1]:.2f}, "
              f"movement={features[i,2]:.2f}, sound={features[i,3]:.2f}")
    
    # Build PyG Data object
    print("\nBuilding graph from numpy arrays...")
    data = build_data_from_numpy(features, adj, node_ids)
    print(f"Graph: {data.x.shape[0]} nodes, {data.edge_index.shape[1]} edges")
    
    # Create model
    print("\nInitializing GraphSAGE model...")
    model = GraphSAGENodePredictor(
        in_channels=4,
        hidden_channels=32,  # Smaller for toy example
        num_layers=2,
        dropout=0.3,
        aggr='mean'
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train model
    print("\nTraining model...")
    print("-" * 40)
    
    # For toy example, use first 2 nodes for training, 1 for validation, 1 for test
    train_idx = torch.tensor([0, 1], dtype=torch.long)
    val_idx = torch.tensor([2], dtype=torch.long)
    test_idx = torch.tensor([3], dtype=torch.long)
    
    model, history = train_model(
        model, data, labels,
        train_idx=train_idx,
        val_idx=val_idx,
        lr=1e-2,  # Higher LR for toy example
        weight_decay=1e-5,
        epochs=100,  # Fewer epochs for toy example
        device=device,
        verbose=True
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    print("-" * 40)
    metrics = evaluate_model(model, data, labels, test_idx, device)
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")
    print(f"Top-5 Recall: {metrics['top_5_recall']:.2%}")
    
    # Get predictions
    print("\nPredicted Priority Scores:")
    print("-" * 40)
    predicted_scores = predict_node_scores(model, data, device)
    for i, (node_id, true_score, pred_score) in enumerate(zip(node_ids, labels, predicted_scores)):
        print(f"Node {i} ({node_id}): True={true_score:.3f}, Predicted={pred_score:.3f}")
    
    # Convert to edge weights for Dijkstra
    print("\nConverting to edge weights for pathfinding...")
    print("-" * 40)
    G = node_scores_to_edge_weights(
        data, predicted_scores, 
        method='avg', alpha=1.0, beta=2.0  # Prioritize rescue urgency
    )
    
    print("Edge weights (lower = easier to traverse):")
    for u, v, data_dict in G.edges(data=True):
        print(f"  Edge {u}-{v}: weight={data_dict['weight']:.3f} "
              f"(priority_cost={data_dict['priority_cost']:.3f})")
    
    # Run Dijkstra from safe zone (node 1) to highest priority (node 3)
    print("\nFinding rescue path from SafeZone to School...")
    source, target = 1, 3
    try:
        path_length, path = nx.single_source_dijkstra(G, source, target)
        print(f"Optimal path: {' -> '.join([f'{p}({node_ids[p]})' for p in path])}")
        print(f"Total cost: {path_length:.3f}")
    except nx.NetworkXNoPath:
        print(f"No path found from {source} to {target}")
    
    # Save model
    print("\nSaving model...")
    save_model(model, 'gnn_model.pth')
    
    # Test loading
    print("Testing model loading...")
    new_model = GraphSAGENodePredictor(
        in_channels=4,
        hidden_channels=32,
        num_layers=2,
        dropout=0.3,
        aggr='mean'
    )
    new_model = load_model(new_model, 'gnn_model.pth', device)
    
    # Verify loaded model works
    new_scores = predict_node_scores(new_model, data, device)
    print(f"Loaded model predictions match: {np.allclose(predicted_scores, new_scores)}")
    
    print("\n" + "=" * 70)
    print("Integration Guide:")
    print("-" * 40)
    print("""
To integrate with existing pipeline:

1. Load real data from CSV:
   ```python
   from load import load_node_features, load_adjacency_matrix, normalize_features
   
   nodes, features = load_node_features('node_features.csv')
   _, adj_matrix = load_adjacency_matrix('adjacency.csv')
   features = normalize_features(features)
   ```

2. Create Data object:
   ```python
   data = build_data_from_numpy(features, adj_matrix, nodes)
   ```

3. Train with real labels:
   ```python
   model = GraphSAGENodePredictor(in_channels=4)
   model, history = train_model(model, data, labels)
   ```

4. Use for pathfinding:
   ```python
   scores = predict_node_scores(model, data)
   G = node_scores_to_edge_weights(data, scores)
   path_length, path = nx.single_source_dijkstra(G, source, target)
   ```

Hyperparameter suggestions:
- hidden_channels: 64-128 for real graphs
- num_layers: 2-3 (more for larger graphs)
- dropout: 0.3-0.5 (higher for overfitting)
- lr: 1e-3 to 1e-4
- epochs: 200-500
- alpha/beta in edge weights: tune based on priority vs distance importance
""")
    print("=" * 70)
