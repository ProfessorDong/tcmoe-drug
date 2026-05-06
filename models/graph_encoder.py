"""
Graph Neural Network Encoder for Molecular Graphs
Implements Graph Convolution and Graph Attention Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def scatter_add(src, index, dim=0, dim_size=None):
    """Native PyTorch replacement for torch_scatter.scatter_add."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    # Expand index to match src shape
    if src.dim() > 1 and index.dim() == 1:
        shape = [1] * src.dim()
        shape[dim] = -1
        index_expanded = index.view(*shape).expand_as(src)
    else:
        index_expanded = index
    out = torch.zeros(*[dim_size if d == dim else src.size(d) for d in range(src.dim())],
                       dtype=src.dtype, device=src.device)
    return out.scatter_add(dim, index_expanded, src)


def scatter_mean(src, index, dim=0, dim_size=None):
    """Native PyTorch replacement for torch_scatter.scatter_mean."""
    sums = scatter_add(src, index, dim=dim, dim_size=dim_size)
    ones = torch.ones_like(src)
    counts = scatter_add(ones, index, dim=dim, dim_size=dim_size)
    return sums / counts.clamp(min=1)


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer with edge features support.
    Aggregates neighbor information using learnable weights.
    """

    def __init__(self, in_features, out_features, edge_features=0,
                 activation='relu', dropout=0.0, bias=True):
        super(GraphConvLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features

        # Node transformation
        self.node_linear = nn.Linear(in_features, out_features, bias=bias)

        # Neighbor transformation
        self.neighbor_linear = nn.Linear(in_features, out_features, bias=False)

        # Edge transformation (if edge features provided)
        if edge_features > 0:
            self.edge_linear = nn.Linear(edge_features, out_features, bias=False)
        else:
            self.edge_linear = None

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(out_features)

        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Identity()

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.node_linear.weight)
        nn.init.xavier_uniform_(self.neighbor_linear.weight)
        if self.edge_linear is not None:
            nn.init.xavier_uniform_(self.edge_linear.weight)
        if self.node_linear.bias is not None:
            nn.init.zeros_(self.node_linear.bias)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Forward pass for graph convolution.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features] (optional)
            batch: Batch assignment for nodes [num_nodes] (optional)

        Returns:
            Updated node features [num_nodes, out_features]
        """
        row, col = edge_index

        # Self transformation
        out = self.node_linear(x)

        # Neighbor aggregation
        neighbor_features = self.neighbor_linear(x[col])

        # Add edge features if available
        if edge_attr is not None and self.edge_linear is not None:
            edge_features = self.edge_linear(edge_attr)
            neighbor_features = neighbor_features + edge_features

        # Sum aggregation
        aggr = scatter_add(neighbor_features, row, dim=0, dim_size=x.size(0))

        out = out + aggr
        out = self.batch_norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        return out


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) with multi-head attention.
    Computes attention coefficients for neighbor aggregation.
    """

    def __init__(self, in_features, out_features, heads=4,
                 concat=True, dropout=0.0, negative_slope=0.2):
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope

        # Linear transformation for each head
        self.linear = nn.Linear(in_features, heads * out_features, bias=False)

        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_features))

        self.dropout = nn.Dropout(dropout)

        if concat:
            self.output_dim = heads * out_features
        else:
            self.output_dim = out_features

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index, return_attention=False):
        """
        Forward pass with multi-head attention.

        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            return_attention: Whether to return attention weights

        Returns:
            Updated node features and optionally attention weights
        """
        H, C = self.heads, self.out_features

        # Linear transformation
        x = self.linear(x).view(-1, H, C)  # [N, H, C]

        row, col = edge_index

        # Compute attention coefficients
        alpha_src = (x * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # [N, H]

        # Edge attention
        alpha = alpha_src[row] + alpha_dst[col]  # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Softmax normalization per node
        alpha = self._softmax(alpha, row, num_nodes=x.size(0))
        alpha = self.dropout(alpha)

        # Weighted aggregation
        out = x[col] * alpha.unsqueeze(-1)  # [E, H, C]
        out = scatter_add(out, row, dim=0, dim_size=x.size(0))  # [N, H, C]

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if return_attention:
            return out, alpha
        return out

    def _softmax(self, alpha, index, num_nodes):
        """Sparse softmax normalization."""
        alpha = alpha - scatter_add(alpha, index, dim=0, dim_size=num_nodes)[index]
        alpha = alpha.exp()
        alpha_sum = scatter_add(alpha, index, dim=0, dim_size=num_nodes)[index] + 1e-16
        return alpha / alpha_sum


class GraphPooling(nn.Module):
    """
    Graph-level pooling to obtain molecular representation.
    Supports sum, mean, max, and attention-based pooling.
    """

    def __init__(self, in_features, pooling_type='attention', hidden_dim=128):
        super(GraphPooling, self).__init__()

        self.pooling_type = pooling_type
        self.in_features = in_features

        if pooling_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        elif pooling_type == 'set2set':
            self.lstm = nn.LSTM(in_features, in_features, batch_first=True)
            self.num_steps = 3

    def forward(self, x, batch, return_attention=False):
        """
        Pool node features to graph-level representation.

        Args:
            x: Node features [num_nodes, features]
            batch: Batch assignment [num_nodes]
            return_attention: Return attention weights (for attention pooling)

        Returns:
            Graph-level features [batch_size, features]
        """
        if self.pooling_type == 'sum':
            return scatter_add(x, batch, dim=0)
        elif self.pooling_type == 'mean':
            return scatter_mean(x, batch, dim=0)
        elif self.pooling_type == 'max':
            dim_size = batch.max().item() + 1
            index_expanded = batch.unsqueeze(-1).expand_as(x)
            out = torch.full((dim_size, x.size(1)), float('-inf'), dtype=x.dtype, device=x.device)
            out.scatter_reduce_(0, index_expanded, x, reduce='amax')
            return out
        elif self.pooling_type == 'attention':
            # Compute attention weights
            att_weights = self.attention(x).squeeze(-1)
            att_weights = self._softmax_by_batch(att_weights, batch)

            # Weighted sum
            out = scatter_add(x * att_weights.unsqueeze(-1), batch, dim=0)

            if return_attention:
                return out, att_weights
            return out
        else:
            return scatter_add(x, batch, dim=0)

    def _softmax_by_batch(self, x, batch):
        """Softmax normalized within each graph."""
        x = x - scatter_add(x, batch, dim=0)[batch]
        x = x.exp()
        x_sum = scatter_add(x, batch, dim=0)[batch] + 1e-16
        return x / x_sum


class GraphEncoder(nn.Module):
    """
    Complete Graph Neural Network Encoder for molecular graphs.
    Combines multiple graph conv/attention layers with pooling.
    """

    def __init__(self, node_features=75, edge_features=6, hidden_dims=[128, 256, 128],
                 output_dim=128, layer_type='gat', num_heads=4, dropout=0.1,
                 pooling='attention'):
        super(GraphEncoder, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.layer_type = layer_type

        # Build graph layers
        self.layers = nn.ModuleList()
        in_dim = node_features

        for i, hidden_dim in enumerate(hidden_dims):
            if layer_type == 'gcn':
                self.layers.append(
                    GraphConvLayer(
                        in_dim, hidden_dim,
                        edge_features=edge_features if i == 0 else 0,
                        dropout=dropout
                    )
                )
                in_dim = hidden_dim
            elif layer_type == 'gat':
                concat = (i < len(hidden_dims) - 1)
                self.layers.append(
                    GraphAttentionLayer(
                        in_dim, hidden_dim // num_heads if concat else hidden_dim,
                        heads=num_heads, concat=concat, dropout=dropout
                    )
                )
                in_dim = hidden_dim if concat else hidden_dim

        # Pooling layer
        self.pooling = GraphPooling(in_dim, pooling_type=pooling)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(in_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

        # Store attention weights for interpretability
        self.attention_weights = None

    def forward(self, data, return_attention=False, return_node_embeddings=False):
        """
        Encode molecular graph to fixed-size representation.

        Args:
            data: Graph data with x, edge_index, edge_attr, batch
            return_attention: Whether to return attention weights
            return_node_embeddings: Whether to return pre-pooling node
                embeddings (for pre-pooling cross-attention)

        Returns:
            Graph representation [batch_size, output_dim]
            If return_node_embeddings: also returns (node_embeddings, batch_vec)
                where node_embeddings is [total_nodes, hidden_dim] BEFORE pooling
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        attention_weights = []

        # Apply graph layers
        for i, layer in enumerate(self.layers):
            if self.layer_type == 'gcn':
                x = layer(x, edge_index, edge_attr if i == 0 else None, batch)
            elif self.layer_type == 'gat':
                if return_attention:
                    x, att = layer(x, edge_index, return_attention=True)
                    attention_weights.append(att)
                else:
                    x = layer(x, edge_index)

        # x now holds per-node embeddings [total_nodes, hidden_dim] BEFORE pooling
        node_embeddings = x  # save for cross-attention

        # Pool to graph level
        if return_attention:
            graph_repr, pool_att = self.pooling(x, batch, return_attention=True)
            attention_weights.append(pool_att)
        else:
            graph_repr = self.pooling(x, batch)

        # Project to output dimension
        output = self.output_proj(graph_repr)

        if return_attention:
            self.attention_weights = attention_weights
            if return_node_embeddings:
                return output, attention_weights, node_embeddings, batch
            return output, attention_weights

        if return_node_embeddings:
            return output, node_embeddings, batch

        return output

    def get_node_embeddings(self, data):
        """Get intermediate node embeddings for visualization."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        embeddings = [x]

        for i, layer in enumerate(self.layers):
            if self.layer_type == 'gcn':
                x = layer(x, edge_index, edge_attr if i == 0 else None)
            else:
                x = layer(x, edge_index)
            embeddings.append(x)

        return embeddings
