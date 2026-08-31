import torch
from torch import nn

from .gnn import GATClassifier
from .losses import g_loss


class GemmaGATModel(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int, classes: int = 2, heads: int = 4):
        super().__init__()
        self.gat = GATClassifier(text_dim, hidden_dim, classes, heads)

    def loss(self, causal, residual, edge_index, label, alpha, beta):
        causal_logits = self.gat(causal, edge_index)
        residual_logits = self.gat(residual, edge_index)
        return g_loss(causal_logits, residual_logits, label, alpha, beta)

    def predict(self, features, edge_index):
        return self.gat(features, edge_index)