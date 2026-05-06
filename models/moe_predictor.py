"""
Multi-View Mixture-of-Experts Predictor for cancer-targeted pIC50 prediction.

Architecture
------------
Four representation experts, each a small MLP head on a distinct molecular view:
    1. Morgan ECFP4   (1024-bit, radius 2)
    2. Dual encoder   (128-dim fused embedding)
    3. MACCS keys     (167-bit pharmacophore)
    4. RDKit physchem (24 standardized descriptors)

A router takes a pooled multi-view input concatenated with a learnable
target-ID embedding and outputs a sparse top-K softmax over experts. Top-K
selection uses noisy gating (Gaussian noise on logits during training).
The final pIC50 prediction is the gate-weighted sum of expert outputs.

A load-balancing auxiliary loss (Shazeer-style coefficient of variation
of per-expert routing mass) is added to discourage gate collapse.

This is the only architectural change required to fix the cross-target
selectivity collapse documented in the JBHI paper §V-G.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Expert head
# ---------------------------------------------------------------------------

class ExpertHead(nn.Module):
    """Small MLP regressor for one view."""

    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Router with target conditioning
# ---------------------------------------------------------------------------

class TargetConditionedRouter(nn.Module):
    """Gating network that mixes views.

    Inputs:
        view_summary: a low-dimensional pooled summary of each view, concatenated.
        target_id:    integer target index in [0, n_targets).

    Outputs:
        gates:        sparse top-K softmax over experts, shape [B, n_experts].
        raw_logits:   pre-softmax logits (for load-balance loss).
        routing_freq: per-expert mean routing mass over the batch.
    """

    def __init__(
        self,
        summary_dim: int,
        n_experts: int,
        n_targets: int,
        target_emb_dim: int = 16,
        hidden: int = 64,
        top_k: int = 2,
        noise_std: float = 1.0,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.noise_std = noise_std

        self.target_emb = nn.Embedding(n_targets, target_emb_dim)
        nn.init.normal_(self.target_emb.weight, std=0.1)

        self.router = nn.Sequential(
            nn.Linear(summary_dim + target_emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_experts),
        )
        # Trainable noise-magnitude gates (Shazeer noisy gating)
        self.noise_gates = nn.Linear(summary_dim + target_emb_dim, n_experts)

    def forward(
        self,
        view_summary: torch.Tensor,
        target_id: torch.Tensor,
        training: bool | None = None,
    ):
        if training is None:
            training = self.training
        emb = self.target_emb(target_id)
        feat = torch.cat([view_summary, emb], dim=-1)
        logits = self.router(feat)

        if training and self.noise_std > 0:
            noise_scale = F.softplus(self.noise_gates(feat))
            noise = torch.randn_like(logits) * noise_scale * self.noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        # Top-K selection
        if self.top_k < self.n_experts:
            topk_vals, topk_idx = noisy_logits.topk(self.top_k, dim=-1)
            mask = torch.full_like(noisy_logits, float('-inf'))
            mask.scatter_(-1, topk_idx, topk_vals)
            gates = F.softmax(mask, dim=-1)
        else:
            gates = F.softmax(noisy_logits, dim=-1)

        # Per-expert routing mass for load balance
        routing_freq = gates.mean(dim=0)  # [n_experts]
        return gates, logits, routing_freq


# ---------------------------------------------------------------------------
# View summary encoder (compresses each view to a small vector for routing)
# ---------------------------------------------------------------------------

class ViewSummary(nn.Module):
    """Compresses each view to a small summary used as router input."""

    def __init__(self, view_dims: list[int], summary_dim: int = 32):
        super().__init__()
        self.summary_dim = summary_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, summary_dim),
                nn.LayerNorm(summary_dim),
                nn.Tanh(),
            )
            for d in view_dims
        ])

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        # views[i] has shape [B, view_dims[i]]
        summaries = [h(v) for h, v in zip(self.heads, views)]
        return torch.cat(summaries, dim=-1)  # [B, n_views * summary_dim]


# ---------------------------------------------------------------------------
# Full MoE predictor
# ---------------------------------------------------------------------------

class MoEPredictor(nn.Module):
    """Multi-view MoE predictor with target-conditioned sparse routing.

    Args:
        view_dims:     list of input dimensions per view (e.g., [1024, 128, 167, 24]).
        view_names:    optional names for logging (must match view_dims length).
        n_targets:     number of distinct biological targets that share this MoE.
        expert_hidden: hidden width of each expert MLP.
        top_k:         number of experts active per molecule (sparse routing).
        target_emb_dim:learnable target embedding dimension.
        summary_dim:   dimension each view is compressed to for the router.
        load_balance:  load-balance coefficient (Shazeer CV(routing_freq)^2).
        dropout:       dropout in expert heads.
    """

    def __init__(
        self,
        view_dims: list[int],
        view_names: list[str] | None = None,
        n_targets: int = 4,
        expert_hidden: int = 128,
        top_k: int = 2,
        target_emb_dim: int = 16,
        summary_dim: int = 32,
        load_balance: float = 0.01,
        dropout: float = 0.2,
        noise_std: float = 1.0,
    ):
        super().__init__()
        self.view_dims = list(view_dims)
        self.n_views = len(view_dims)
        self.view_names = view_names or [f'view{i}' for i in range(self.n_views)]
        self.n_targets = n_targets
        self.top_k = top_k
        self.load_balance = load_balance
        self.target_emb_dim = target_emb_dim

        # Per-view experts
        self.experts = nn.ModuleList([
            ExpertHead(d, hidden=expert_hidden, dropout=dropout)
            for d in view_dims
        ])

        # Per-target output bias (helps the router separate target activity ranges)
        self.target_bias = nn.Embedding(n_targets, 1)
        nn.init.zeros_(self.target_bias.weight)

        # View summaries -> router
        self.view_summary = ViewSummary(view_dims, summary_dim=summary_dim)
        summary_total = self.n_views * summary_dim
        self.router = TargetConditionedRouter(
            summary_dim=summary_total,
            n_experts=self.n_views,
            n_targets=n_targets,
            target_emb_dim=target_emb_dim,
            hidden=64,
            top_k=top_k,
            noise_std=noise_std,
        )

    def forward(
        self,
        views: list[torch.Tensor],
        target_id: torch.Tensor,
        return_aux: bool = False,
    ):
        """
        Args:
            views:     list of tensors, one per view, each [B, view_dims[i]].
            target_id: [B] int64 tensor of target indices.
            return_aux:also return gates, expert outputs, and load-balance loss.
        """
        assert len(views) == self.n_views, (
            f'Expected {self.n_views} views, got {len(views)}'
        )

        # Per-expert predictions
        expert_preds = torch.stack([e(v) for e, v in zip(self.experts, views)],
                                   dim=-1)  # [B, n_experts]

        # Routing
        summary = self.view_summary(views)
        gates, raw_logits, routing_freq = self.router(summary, target_id)

        # Mixture
        mix = (gates * expert_preds).sum(dim=-1)  # [B]
        bias = self.target_bias(target_id).squeeze(-1)
        y = mix + bias

        if not return_aux:
            return y

        # Load-balance loss: CV^2 of per-expert mass.
        # See Shazeer et al. 2017 (Outrageously Large Neural Networks).
        eps = 1e-8
        mean = routing_freq.mean()
        var = ((routing_freq - mean) ** 2).mean()
        cv2 = var / (mean ** 2 + eps)
        lb_loss = self.load_balance * cv2

        aux = {
            'gates': gates,
            'expert_preds': expert_preds,
            'logits': raw_logits,
            'routing_freq': routing_freq,
            'load_balance_loss': lb_loss,
        }
        return y, aux

    @torch.no_grad()
    def expert_utilization(self, views, target_id):
        """Mean per-expert routing mass over a batch (no gradients)."""
        self.eval()
        summary = self.view_summary(views)
        gates, _, _ = self.router(summary, target_id, training=False)
        return gates.mean(dim=0).cpu().numpy()


# ---------------------------------------------------------------------------
# Lightweight factory
# ---------------------------------------------------------------------------

def make_default_moe(
    encoder_dim: int = 128,
    n_targets: int = 4,
    top_k: int = 2,
    target_emb_dim: int = 16,
    expert_hidden: int = 128,
    load_balance: float = 0.01,
    dropout: float = 0.2,
):
    """Construct an MoE predictor with the default 4-view configuration.

    The view order is fixed as (Morgan, Encoder, MACCS, Descriptors). Make
    sure the view tensors are passed to ``forward`` in this order.
    """
    return MoEPredictor(
        view_dims=[1024, encoder_dim, 167, 24],
        view_names=['morgan', 'encoder', 'maccs', 'descriptors'],
        n_targets=n_targets,
        expert_hidden=expert_hidden,
        top_k=top_k,
        target_emb_dim=target_emb_dim,
        load_balance=load_balance,
        dropout=dropout,
    )
