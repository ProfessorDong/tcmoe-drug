"""
Property Predictor Module
Multi-task property prediction with uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPHead(nn.Module):
    """Multi-layer perceptron prediction head."""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1,
                 activation='relu', output_activation=None):
        super(MLPHead, self).__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer for uncertainty estimation.
    Uses weight uncertainty for epistemic uncertainty.
    """

    def __init__(self, in_features, out_features, prior_std=1.0):
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Weight mean and log variance
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.Tensor(out_features, in_features))

        # Bias mean and log variance
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_var = nn.Parameter(torch.Tensor(out_features))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_log_var, -5)
        nn.init.zeros_(self.bias_mu)
        nn.init.constant_(self.bias_log_var, -5)

    def forward(self, x, sample=True):
        if sample and self.training:
            # Sample weights using reparameterization trick
            weight_std = torch.exp(0.5 * self.weight_log_var)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

            bias_std = torch.exp(0.5 * self.bias_log_var)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """Compute KL divergence from prior."""
        weight_var = torch.exp(self.weight_log_var)
        bias_var = torch.exp(self.bias_log_var)

        kl_weight = 0.5 * (
            weight_var / (self.prior_std ** 2)
            + (self.weight_mu ** 2) / (self.prior_std ** 2)
            - 1
            - self.weight_log_var
            + 2 * np.log(self.prior_std)
        ).sum()

        kl_bias = 0.5 * (
            bias_var / (self.prior_std ** 2)
            + (self.bias_mu ** 2) / (self.prior_std ** 2)
            - 1
            - self.bias_log_var
            + 2 * np.log(self.prior_std)
        ).sum()

        return kl_weight + kl_bias


class PropertyPredictor(nn.Module):
    """
    Multi-task Property Predictor with optional uncertainty estimation.

    Supports:
    - Single and multi-task prediction
    - Classification and regression
    - Bayesian uncertainty estimation
    - One-shot/few-shot learning via metric learning
    """

    def __init__(self, input_dim, task_configs, hidden_dims=[256, 128],
                 dropout=0.1, use_bayesian=False, shared_layers=True):
        """
        Args:
            input_dim: Input feature dimension
            task_configs: Dict mapping task_name to {'type': 'classification'/'regression',
                                                      'num_classes': int (for classification)}
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            use_bayesian: Use Bayesian layers for uncertainty
            shared_layers: Share hidden layers across tasks
        """
        super(PropertyPredictor, self).__init__()

        self.input_dim = input_dim
        self.task_configs = task_configs
        self.use_bayesian = use_bayesian
        self.shared_layers = shared_layers

        # Shared layers
        if shared_layers:
            self.shared = self._build_mlp(input_dim, hidden_dims[:-1], dropout)
            task_input_dim = hidden_dims[-2] if len(hidden_dims) > 1 else input_dim
        else:
            self.shared = nn.Identity()
            task_input_dim = input_dim

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            if config['type'] == 'classification':
                output_dim = config.get('num_classes', 2)
            else:
                output_dim = 1

            if use_bayesian:
                head = nn.Sequential(
                    BayesianLinear(task_input_dim, hidden_dims[-1]),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    BayesianLinear(hidden_dims[-1], output_dim)
                )
            else:
                head = MLPHead(
                    task_input_dim, [hidden_dims[-1]], output_dim, dropout
                )

            self.task_heads[task_name] = head

    def _build_mlp(self, input_dim, hidden_dims, dropout):
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        return nn.Sequential(*layers)

    def forward(self, x, task_name=None):
        """
        Forward pass.

        Args:
            x: Input features [batch, input_dim]
            task_name: Specific task to predict (None = all tasks)

        Returns:
            Dict of predictions per task
        """
        # Shared representation
        shared_repr = self.shared(x)

        # Task predictions
        predictions = {}

        if task_name is not None:
            predictions[task_name] = self.task_heads[task_name](shared_repr)
        else:
            for name, head in self.task_heads.items():
                predictions[name] = head(shared_repr)

        return predictions

    def predict_with_uncertainty(self, x, task_name, n_samples=30):
        """
        Make predictions with uncertainty estimation.

        Args:
            x: Input features [batch, input_dim]
            task_name: Task to predict
            n_samples: Number of MC samples

        Returns:
            mean: Mean prediction
            std: Uncertainty (standard deviation)
        """
        self.train()  # Enable dropout

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, task_name)[task_name]
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        self.eval()
        return mean, std

    def get_kl_loss(self):
        """Get KL divergence loss for Bayesian layers."""
        if not self.use_bayesian:
            return 0.0

        kl_loss = 0.0
        for head in self.task_heads.values():
            for module in head.modules():
                if isinstance(module, BayesianLinear):
                    kl_loss += module.kl_divergence()

        return kl_loss


class MetricLearningPredictor(nn.Module):
    """
    Metric Learning based predictor for one-shot/few-shot learning.
    Uses cosine similarity with support set for predictions.
    """

    def __init__(self, input_dim, embedding_dim=128, temperature=1.0):
        super(MetricLearningPredictor, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.temperature = temperature

    def forward(self, query, support_x, support_y):
        """
        One-shot prediction using support set.

        Args:
            query: Query features [batch, input_dim]
            support_x: Support set features [n_support, input_dim]
            support_y: Support set labels [n_support]

        Returns:
            Predicted probabilities [batch]
        """
        # Embed
        query_emb = self.embedding(query)  # [batch, emb_dim]
        support_emb = self.embedding(support_x)  # [n_support, emb_dim]

        # Normalize for cosine similarity
        query_emb = F.normalize(query_emb, dim=-1)
        support_emb = F.normalize(support_emb, dim=-1)

        # Compute similarity
        similarity = torch.mm(query_emb, support_emb.t()) / self.temperature
        attention = F.softmax(similarity, dim=-1)

        # Weighted sum of support labels
        support_y = support_y.float()
        predictions = torch.mv(attention, support_y)

        return predictions

    def compute_prototypes(self, support_x, support_y):
        """Compute class prototypes for prototypical networks."""
        support_emb = self.embedding(support_x)
        support_emb = F.normalize(support_emb, dim=-1)

        classes = torch.unique(support_y)
        prototypes = []

        for c in classes:
            mask = support_y == c
            prototype = support_emb[mask].mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes), classes


class OneShotPredictor(nn.Module):
    """
    One-shot predictor combining encoder with metric learning.
    Designed for few-shot property prediction.
    """

    def __init__(self, encoder, embedding_dim=128, n_support=20):
        super(OneShotPredictor, self).__init__()

        self.encoder = encoder
        self.n_support = n_support

        # Get encoder output dimension
        encoder_dim = encoder.output_dim

        # Metric learning layer
        self.metric = MetricLearningPredictor(encoder_dim, embedding_dim)

    def forward(self, query_data, support_data, support_labels,
                query_smiles=None, support_smiles=None,
                query_lengths=None, support_lengths=None):
        """
        One-shot prediction.

        Args:
            query_data: Query graph data or features
            support_data: Support graph data or features
            support_labels: Support set labels
            *_smiles: SMILES data for dual encoder
            *_lengths: SMILES lengths

        Returns:
            Predictions for query set
        """
        # Encode query
        if hasattr(self.encoder, 'smiles_encoder'):  # Dual encoder
            query_repr = self.encoder(
                query_data, query_smiles, query_lengths
            )
            support_repr = self.encoder(
                support_data, support_smiles, support_lengths
            )
        else:  # Single encoder
            query_repr = self.encoder(query_data)
            support_repr = self.encoder(support_data)

        # Metric learning prediction
        predictions = self.metric(query_repr, support_repr, support_labels)

        return predictions


class DualEncoderPredictor(nn.Module):
    """Property predictor that uses dual encoder embeddings (f) instead of
    or in addition to Morgan fingerprints.

    This closes the generator-predictor representational gap: the same
    learned chemistry that drives generation also informs the reward signal.

    Args:
        encoder_dim: Dimension of dual encoder output f (default 128).
        fp_dim: Dimension of Morgan fingerprint (default 1024, 0 to disable).
        hidden_dims: Hidden layer dimensions for the predictor MLP.
        dropout: Dropout rate.
    """

    def __init__(self, encoder_dim=128, fp_dim=1024, hidden_dims=[512, 256],
                 dropout=0.1):
        super(DualEncoderPredictor, self).__init__()
        self.encoder_dim = encoder_dim
        self.fp_dim = fp_dim
        input_dim = encoder_dim + fp_dim

        task_configs = {'pIC50': {'type': 'regression'}}
        self.predictor = PropertyPredictor(
            input_dim=input_dim,
            task_configs=task_configs,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, encoder_output, morgan_fp=None, task_name='pIC50'):
        """
        Args:
            encoder_output: Dual encoder fused representation [B, encoder_dim].
            morgan_fp: Optional Morgan fingerprint [B, fp_dim].
            task_name: Task name for multi-task predictor.
        """
        if morgan_fp is not None and self.fp_dim > 0:
            x = torch.cat([encoder_output, morgan_fp], dim=-1)
        else:
            x = encoder_output
        return self.predictor(x, task_name=task_name)
