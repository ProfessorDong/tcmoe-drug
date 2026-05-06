"""
Dual Encoder: Combines Graph and SMILES representations
Implements attention-based fusion of dual molecular representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_encoder import GraphEncoder
from .smiles_encoder import SMILESEncoder


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between graph and SMILES representations.
    Allows each modality to attend to the other.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super(CrossModalAttention, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key, value, return_attention=False):
        """
        Cross-modal attention.

        Args:
            query: Query tensor [batch, query_len, dim]
            key: Key tensor [batch, key_len, dim]
            value: Value tensor [batch, key_len, dim]
            return_attention: Return attention weights

        Returns:
            Attended output [batch, query_len, dim]
        """
        batch_size = query.size(0)

        # Project
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        out = self.out_proj(out)

        if return_attention:
            return out, attn
        return out


class FusionLayer(nn.Module):
    """
    Fusion layer to combine graph and SMILES representations.
    Supports multiple fusion strategies.
    """

    def __init__(self, dim, fusion_type='attention', dropout=0.1):
        super(FusionLayer, self).__init__()

        self.fusion_type = fusion_type
        self.dim = dim

        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            )
            self.transform = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
                nn.ReLU()
            )
        elif fusion_type == 'attention':
            self.cross_attn_g2s = CrossModalAttention(dim, dropout=dropout)
            self.cross_attn_s2g = CrossModalAttention(dim, dropout=dropout)
            # Learned gate to modulate modality contributions
            self.gate_proj = nn.Linear(dim * 2, dim)
            self.fusion = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == 'bilinear':
            self.bilinear = nn.Bilinear(dim, dim, dim)
            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, graph_repr, smiles_repr, return_attention=False):
        """
        Fuse graph and SMILES representations.

        Args:
            graph_repr: Graph representation [batch, dim]
            smiles_repr: SMILES representation [batch, dim]
            return_attention: Return attention weights (for attention fusion)

        Returns:
            Fused representation [batch, dim]
        """
        if self.fusion_type == 'concat':
            combined = torch.cat([graph_repr, smiles_repr], dim=-1)
            return self.fusion(combined)

        elif self.fusion_type == 'gate':
            combined = torch.cat([graph_repr, smiles_repr], dim=-1)
            gate = self.gate(combined)
            transformed = self.transform(combined)
            return gate * graph_repr + (1 - gate) * smiles_repr + transformed

        elif self.fusion_type == 'attention':
            # Expand dims for attention
            g = graph_repr.unsqueeze(1)
            s = smiles_repr.unsqueeze(1)

            # Cross-modal attention
            if return_attention:
                g_attended, attn_g2s = self.cross_attn_g2s(g, s, s, return_attention=True)
                s_attended, attn_s2g = self.cross_attn_s2g(s, g, g, return_attention=True)
            else:
                g_attended = self.cross_attn_g2s(g, s, s)
                s_attended = self.cross_attn_s2g(s, g, g)

            # Squeeze
            g_attended = g_attended.squeeze(1)
            s_attended = s_attended.squeeze(1)

            # Gated fusion: λ modulates relative modality contribution
            lam = torch.sigmoid(self.gate_proj(
                torch.cat([g_attended, s_attended], dim=-1)
            ))
            fused_input = lam * g_attended + (1 - lam) * s_attended
            fused = self.fusion(fused_input)

            if return_attention:
                return fused, {'g2s': attn_g2s, 's2g': attn_s2g}
            return fused

        elif self.fusion_type == 'bilinear':
            fused = self.bilinear(graph_repr, smiles_repr)
            return self.layer_norm(F.relu(fused))


class PrePoolingCrossAttention(nn.Module):
    """Cross-attention applied BEFORE pooling, over sequences of node/token
    embeddings.  This yields real multi-head attention (multiple Q/K pairs)
    instead of the degenerate scalar case that occurs with single-vector
    inputs.

    Args:
        graph_dim:  Dimension of per-node graph embeddings (e.g. 128).
        seq_dim:    Dimension of per-token sequence embeddings (e.g. 512).
        proj_dim:   Common projection dimension for Q/K/V (default 128).
        num_heads:  Number of attention heads (default 4).
        dropout:    Attention dropout rate.
    """

    def __init__(self, graph_dim, seq_dim, proj_dim=128, num_heads=4,
                 dropout=0.1):
        super().__init__()
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.head_dim = proj_dim // num_heads
        assert proj_dim % num_heads == 0
        self.scale = self.head_dim ** -0.5

        # Project sequence embeddings to same dim as graph
        self.seq_proj = nn.Linear(seq_dim, proj_dim)

        # Graph-to-Sequence attention (nodes query tokens)
        self.g2s_q = nn.Linear(graph_dim, proj_dim)
        self.g2s_k = nn.Linear(proj_dim, proj_dim)
        self.g2s_v = nn.Linear(proj_dim, proj_dim)
        self.g2s_out = nn.Linear(proj_dim, graph_dim)

        # Sequence-to-Graph attention (tokens query nodes)
        self.s2g_q = nn.Linear(proj_dim, proj_dim)
        self.s2g_k = nn.Linear(graph_dim, proj_dim)
        self.s2g_v = nn.Linear(graph_dim, proj_dim)
        self.s2g_out = nn.Linear(proj_dim, proj_dim)

        self.dropout = nn.Dropout(dropout)

    def _mha(self, Q, K, V, mask=None):
        """Standard multi-head attention.
        Q: [B, Lq, proj_dim], K: [B, Lk, proj_dim], V: [B, Lk, proj_dim]
        mask: [B, 1, 1, Lk] (True = masked position)
        Returns: [B, Lq, proj_dim]
        """
        B, Lq, _ = Q.shape
        Lk = K.shape[1]
        H, D = self.num_heads, self.head_dim

        Q = Q.view(B, Lq, H, D).transpose(1, 2)  # [B, H, Lq, D]
        K = K.view(B, Lk, H, D).transpose(1, 2)
        V = V.view(B, Lk, H, D).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B,H,Lq,Lk]
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, Lq, D]
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.proj_dim)
        return out, attn

    def forward(self, H_graph, graph_batch, H_seq, seq_lengths):
        """Pre-pooling bidirectional cross-attention.

        Args:
            H_graph:     Node embeddings [total_nodes, graph_dim] (ragged).
            graph_batch: Batch assignment [total_nodes] (int, 0..B-1).
            H_seq:       Token embeddings [B, max_seq_len, seq_dim].
            seq_lengths: Actual token lengths [B].

        Returns:
            H_graph_att: Cross-attended node embeddings [total_nodes, graph_dim].
            H_seq_att:   Cross-attended token embeddings [B, max_seq_len, proj_dim].
            attn_g2s:    Graph-to-sequence attention weights.
        """
        device = H_graph.device
        B = int(graph_batch.max().item()) + 1
        max_nodes = max(int((graph_batch == b).sum().item()) for b in range(B))
        max_tokens = H_seq.shape[1]

        # Pad graph nodes into [B, max_nodes, graph_dim]
        graph_dim = H_graph.shape[1]
        H_graph_pad = torch.zeros(B, max_nodes, graph_dim, device=device)
        node_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=device)
        for b in range(B):
            idx = (graph_batch == b).nonzero(as_tuple=True)[0]
            n = idx.shape[0]
            H_graph_pad[b, :n] = H_graph[idx]
            node_mask[b, :n] = False  # False = valid

        # Project sequence
        H_seq_proj = self.seq_proj(H_seq)  # [B, max_tokens, proj_dim]

        # Sequence mask
        seq_mask = torch.arange(max_tokens, device=device).unsqueeze(0) >= \
                   seq_lengths.to(device).unsqueeze(1)  # [B, max_tokens], True=pad

        # --- Graph-to-Sequence attention (nodes attend to tokens) ---
        Q_g2s = self.g2s_q(H_graph_pad)   # [B, max_nodes, proj_dim]
        K_g2s = self.g2s_k(H_seq_proj)    # [B, max_tokens, proj_dim]
        V_g2s = self.g2s_v(H_seq_proj)    # [B, max_tokens, proj_dim]
        # mask shape: [B, 1, 1, max_tokens]
        g2s_mask = seq_mask.unsqueeze(1).unsqueeze(2)
        H_graph_att_pad, attn_g2s = self._mha(Q_g2s, K_g2s, V_g2s, mask=g2s_mask)
        H_graph_att_pad = self.g2s_out(H_graph_att_pad)  # [B, max_nodes, graph_dim]
        # Residual connection
        H_graph_att_pad = H_graph_att_pad + H_graph_pad

        # --- Sequence-to-Graph attention (tokens attend to nodes) ---
        Q_s2g = self.s2g_q(H_seq_proj)    # [B, max_tokens, proj_dim]
        K_s2g = self.s2g_k(H_graph_pad)   # [B, max_nodes, proj_dim]
        V_s2g = self.s2g_v(H_graph_pad)   # [B, max_nodes, proj_dim]
        s2g_mask = node_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, max_nodes]
        H_seq_att, attn_s2g = self._mha(Q_s2g, K_s2g, V_s2g, mask=s2g_mask)
        H_seq_att = self.s2g_out(H_seq_att)  # [B, max_tokens, proj_dim]
        # Residual connection
        H_seq_att = H_seq_att + H_seq_proj

        # Unpad graph back to ragged [total_nodes, graph_dim]
        H_graph_att = torch.zeros_like(H_graph)
        for b in range(B):
            idx = (graph_batch == b).nonzero(as_tuple=True)[0]
            n = idx.shape[0]
            H_graph_att[idx] = H_graph_att_pad[b, :n]

        return H_graph_att, H_seq_att, attn_g2s


class DualEncoder(nn.Module):
    """
    Dual Encoder combining Graph Neural Networks and SMILES RNN with
    PRE-POOLING cross-attention and gated fusion.

    The cross-attention operates on per-node graph embeddings (|V|×D) and
    per-token sequence embeddings (m×D) BEFORE pooling, ensuring that the
    multi-head attention has multiple Q/K pairs (non-degenerate).
    """

    def __init__(self, node_features=75, edge_features=6, vocab_size=50,
                 graph_hidden_dims=[128, 256, 128], smiles_hidden_dim=256,
                 output_dim=128, graph_type='gat', num_heads=4, dropout=0.1,
                 fusion_type='attention', pooling='attention'):
        super(DualEncoder, self).__init__()

        self.output_dim = output_dim
        self.fusion_type = fusion_type
        graph_node_dim = graph_hidden_dims[-1]  # dim of node embeddings before pooling
        seq_token_dim = smiles_hidden_dim * 2   # bidirectional GRU

        # Graph encoder (GAT)
        self.graph_encoder = GraphEncoder(
            node_features=node_features,
            edge_features=edge_features,
            hidden_dims=graph_hidden_dims,
            output_dim=output_dim,
            layer_type=graph_type,
            num_heads=num_heads,
            dropout=dropout,
            pooling=pooling
        )

        # SMILES encoder (BiGRU)
        self.smiles_encoder = SMILESEncoder(
            vocab_size=vocab_size,
            hidden_dim=smiles_hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            attention=True
        )

        # Pre-pooling cross-attention (the core architectural contribution)
        self.cross_attention = PrePoolingCrossAttention(
            graph_dim=graph_node_dim,
            seq_dim=seq_token_dim,
            proj_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Attention pooling for cross-attended node embeddings
        self.graph_pool_proj = nn.Sequential(
            nn.Linear(graph_node_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
        self.graph_pool_attn = nn.Sequential(
            nn.Linear(graph_node_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Attention pooling for cross-attended token embeddings
        self.seq_pool_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
        self.seq_pool_attn = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Gated fusion: λ ⊙ g + (1-λ) ⊙ s → f
        self.gate_proj = nn.Linear(output_dim * 2, output_dim)
        self.fusion_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Legacy fusion layer for backward compatibility (concat/bilinear modes)
        self.legacy_fusion = FusionLayer(
            dim=output_dim, fusion_type=fusion_type, dropout=dropout
        )

    def _pool_nodes(self, H, batch_vec):
        """Attention-pool ragged node embeddings to graph-level vector."""
        # H: [total_nodes, D], batch_vec: [total_nodes]
        scores = self.graph_pool_attn(H).squeeze(-1)  # [total_nodes]
        # Per-graph softmax using scatter
        B = int(batch_vec.max().item()) + 1
        pooled = torch.zeros(B, H.shape[1], device=H.device)
        for b in range(B):
            mask = (batch_vec == b)
            w = F.softmax(scores[mask], dim=0)
            pooled[b] = (w.unsqueeze(1) * H[mask]).sum(0)
        return self.graph_pool_proj(pooled)

    def _pool_tokens(self, H, lengths):
        """Attention-pool padded token embeddings to sequence-level vector."""
        # H: [B, L, D]
        B, L, D = H.shape
        scores = self.seq_pool_attn(H).squeeze(-1)  # [B, L]
        mask = torch.arange(L, device=H.device).unsqueeze(0) >= \
               lengths.to(H.device).unsqueeze(1)
        scores = scores.masked_fill(mask, float('-inf'))
        w = F.softmax(scores, dim=1)  # [B, L]
        pooled = torch.bmm(w.unsqueeze(1), H).squeeze(1)  # [B, D]
        return self.seq_pool_proj(pooled)

    def forward(self, graph_data, smiles_data, smiles_lengths=None,
                return_attention=False, return_individual=False):
        """
        Forward pass with pre-pooling cross-attention.

        Args:
            graph_data: Graph batch data (x, edge_index, batch, ...)
            smiles_data: Tokenized SMILES [batch, seq_len]
            smiles_lengths: SMILES sequence lengths [batch]

        Returns:
            Fused representation f ∈ R^{output_dim}
        """
        # --- Get pre-pooling embeddings ---
        # Graph: node embeddings [total_nodes, graph_node_dim]
        _, node_emb, batch_vec = self.graph_encoder(
            graph_data, return_node_embeddings=True
        )
        # Sequence: token embeddings [B, seq_len, hidden*2]
        _, token_emb = self.smiles_encoder.forward_with_sequence(
            smiles_data, smiles_lengths
        )

        if smiles_lengths is None:
            smiles_lengths = torch.full(
                (smiles_data.shape[0],), smiles_data.shape[1],
                device=smiles_data.device
            )

        # --- Pre-pooling cross-attention (the key contribution) ---
        H_graph_att, H_seq_att, attn_g2s = self.cross_attention(
            node_emb, batch_vec, token_emb, smiles_lengths
        )

        # --- Pool cross-attended embeddings ---
        g = self._pool_nodes(H_graph_att, batch_vec)   # [B, output_dim]
        s = self._pool_tokens(H_seq_att, smiles_lengths)  # [B, output_dim]

        # --- Gated fusion ---
        lam = torch.sigmoid(self.gate_proj(torch.cat([g, s], dim=-1)))
        f_gated = lam * g + (1 - lam) * s
        fused = self.fusion_proj(f_gated)

        if return_individual or return_attention:
            outputs = {'fused': fused, 'graph': g, 'smiles': s}
            if return_attention:
                outputs['attention'] = {'g2s': attn_g2s}
            return outputs

        return fused

    def encode_graph_only(self, graph_data, return_attention=False):
        """Encode using only graph modality (no cross-attention)."""
        return self.graph_encoder(graph_data, return_attention=return_attention)

    def encode_smiles_only(self, smiles_data, smiles_lengths=None, return_attention=False):
        """Encode using only SMILES modality (no cross-attention)."""
        return self.smiles_encoder(smiles_data, smiles_lengths, return_attention=return_attention)

    def get_interpretability_info(self):
        """Get stored attention weights for interpretation."""
        return {
            'graph_attention': self.graph_attention,
            'smiles_attention': self.smiles_attention,
            'fusion_attention': self.fusion_attention
        }
