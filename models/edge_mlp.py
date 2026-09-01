"""Simple MLP on edge traffic (no graph structure)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeMLP(nn.Module):
    """MLP that works directly on edge traffic."""
    
    def __init__(self, n_edges=15, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(n_edges, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 2)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, edge_traffic):
        x = F.relu(self.fc1(edge_traffic))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train_edge_mlp(model, train_dataset, val_dataset, epochs=100, device='cpu'):
    """Train EdgeMLP."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Extract edge traffic and labels
    train_X = torch.stack([data.edge_attr[:15, 0] for data in train_dataset])
    train_y = torch.tensor([data.y.item() for data in train_dataset])
    
    val_X = torch.stack([data.edge_attr[:15, 0] for data in val_dataset])
    val_y = torch.tensor([data.y.item() for data in val_dataset])
    
    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_X, val_y), batch_size=32)
    
    # Class weights
    class_counts = torch.bincount(train_y)
    weights = len(train_y) / (2 * class_counts.float())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = F.nll_loss(out, y, weight=weights.to(device))
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                out = model(X)
                preds.extend(out.argmax(dim=1).cpu().numpy())
                labels.extend(y.numpy())
        
        from sklearn.metrics import f1_score
        f1 = f1_score(labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Val F1: {f1:.4f}")
    
    print(f"Best F1: {best_f1:.4f}")
    return model