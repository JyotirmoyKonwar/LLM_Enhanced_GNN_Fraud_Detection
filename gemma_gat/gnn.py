import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, classes: int, heads: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.first = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.second = GATConv(hidden_dim * heads, classes, heads=1, concat=False, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.dropout(x, self.dropout, self.training)
        x = self.first(x, edge_index).elu()
        x = torch.nn.functional.dropout(x, self.dropout, self.training)
        return self.second(x, edge_index)