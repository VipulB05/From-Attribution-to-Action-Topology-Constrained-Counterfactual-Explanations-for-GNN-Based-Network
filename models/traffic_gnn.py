"""Graph Neural Network models for traffic prediction failure."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.loader import DataLoader
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class TrafficGCN(nn.Module):
    """Graph Convolutional Network for failure classification."""
    
    def __init__(self, node_features, hidden_dim=64, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Classification head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


class TrafficGAT(nn.Module):
    """Graph Attention Network for failure classification."""
    
    def __init__(self, node_features, hidden_dim=64, num_layers=3, 
                 heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(node_features, hidden_dim // heads, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 2)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


def compute_class_weights(dataset):
    """Compute class weights for imbalanced dataset."""
    labels = torch.tensor([data.y.item() for data in dataset])
    class_counts = torch.bincount(labels)
    
    # Inverse frequency weighting
    total = len(labels)
    weights = total / (len(class_counts) * class_counts.float())
    
    return weights


def train_gnn(model, train_loader, optimizer, device, class_weights=None):
    """Train for one epoch with class weighting."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        
        # Use weighted loss
        if class_weights is not None:
            weights = class_weights.to(device)
            loss = F.nll_loss(out, data.y.squeeze(), weight=weights)
        else:
            loss = F.nll_loss(out, data.y.squeeze())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze()).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_gnn(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.squeeze())
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y.squeeze()).sum().item()
        total += data.num_graphs
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(data.y.squeeze().cpu().numpy())
        all_probs.extend(torch.exp(out)[:, 1].cpu().numpy())
    
    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def train_model_full(model, train_dataset, val_dataset, 
                    epochs=100, lr=0.001, batch_size=32, device='cpu'):
    """Full training loop with class weighting and better early stopping."""
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Compute class weights
    class_weights = compute_class_weights(train_dataset)
    print(f"    Class weights: {class_weights.tolist()}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        patience=15, 
        factor=0.5
    )
    
    best_val_f1 = 0
    patience_counter = 0
    patience_limit = 30
    best_model_state = None
    
    for epoch in range(epochs):
        train_loss, train_acc = train_gnn(model, train_loader, optimizer, device, class_weights)
        val_results = evaluate_gnn(model, val_loader, device)
        val_loss, val_acc = val_results['loss'], val_results['accuracy']
        
        # Compute F1 for validation
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_results['labels'], val_results['predictions'], zero_division=0)
        
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"    Best validation F1: {best_val_f1:.4f}")
    
    return model