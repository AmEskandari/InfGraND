"""Teacher GNNs (GCN, GAT, GraphSAGE), student MLP, and the Model wrapper."""
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv, GraphConv, SAGEConv


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(h)
            if l != self.num_layers - 1:
                h = self.dropout(F.relu(h))
        return h


class GCN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GraphConv(input_dim, output_dim, activation=activation))
        else:
            self.layers.append(GraphConv(input_dim, hidden_dim, activation=activation))
            for _ in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
            self.layers.append(GraphConv(hidden_dim, output_dim))

    def forward(self, g, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.dropout(h)
        return h


class GAT(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio,
                 activation, num_heads=8, attn_drop=0.3, negative_slope=0.2, residual=False):
        super().__init__()
        hidden_dim //= num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        heads = ([num_heads] * num_layers) + [1]
        self.layers.append(GATConv(input_dim, hidden_dim, heads[0],
                                   dropout_ratio, attn_drop, negative_slope, False, activation))
        for l in range(1, num_layers - 1):
            self.layers.append(GATConv(hidden_dim * heads[l - 1], hidden_dim, heads[l],
                                       dropout_ratio, attn_drop, negative_slope, residual, activation))
        self.layers.append(GATConv(hidden_dim * heads[-2], output_dim, heads[-1],
                                   dropout_ratio, attn_drop, negative_slope, residual, None))

    def forward(self, g, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout_ratio, activation):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout_ratio)
        self.activation = activation
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(SAGEConv(input_dim, output_dim, aggregator_type="gcn"))
        else:
            self.layers.append(SAGEConv(input_dim, hidden_dim, aggregator_type="gcn"))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggregator_type="gcn"))
            self.layers.append(SAGEConv(hidden_dim, output_dim, aggregator_type="gcn"))

    def forward(self, g, feats):
        h = feats
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            if l != self.num_layers - 1:
                h = self.dropout(self.activation(h))
        return h


def build_teacher(name: str, *, input_dim, hidden_dim, output_dim, num_layers,
                  dropout, num_heads=4):
    if name == "GCN":
        return GCN(num_layers, input_dim, hidden_dim, output_dim, dropout, F.relu)
    if name == "GAT":
        return GAT(num_layers, input_dim, hidden_dim, output_dim, dropout, F.relu,
                   num_heads=num_heads)
    if name == "SAGE":
        return GraphSAGE(num_layers, input_dim, hidden_dim, output_dim, dropout, F.relu)
    raise ValueError(f"Unknown teacher {name!r}")


def build_student(*, input_dim, hidden_dim, output_dim, num_layers, dropout):
    return MLP(num_layers, input_dim, hidden_dim, output_dim, dropout)
