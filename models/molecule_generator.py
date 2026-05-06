"""
Molecule Generator for de novo drug design
Combines Stack-Augmented RNN with RL training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from rdkit import Chem


class MoleculeGenerator(nn.Module):
    """
    Molecule Generator using Stack-Augmented RNN.
    Supports both supervised pretraining and RL fine-tuning.
    """

    def __init__(self, vocab_size, hidden_size=512, embedding_dim=128,
                 n_layers=2, has_stack=True, stack_width=100, stack_depth=100,
                 dropout=0.1, use_cuda=True):
        super(MoleculeGenerator, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.has_stack = has_stack
        self.stack_width = stack_width
        self.stack_depth = stack_depth
        self.use_cuda = use_cuda and torch.cuda.is_available()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Stack augmentation
        if has_stack:
            self.stack_controls = nn.Linear(hidden_size, 3)
            self.stack_input = nn.Linear(hidden_size, stack_width)
            rnn_input_size = embedding_dim + stack_width
        else:
            rnn_input_size = embedding_dim

        # GRU layers
        self.gru = nn.GRU(
            rnn_input_size, hidden_size, n_layers,
            batch_first=False, dropout=dropout if n_layers > 1 else 0
        )

        # Output layer
        self.output = nn.Linear(hidden_size, vocab_size)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        if self.use_cuda:
            self.cuda()

    def forward(self, inp, hidden, stack=None):
        """
        Single step forward pass.

        Args:
            inp: Input token [1] or [batch]
            hidden: Hidden state [n_layers, batch, hidden_size]
            stack: Stack state [batch, stack_depth, stack_width]

        Returns:
            logits: Output logits [1, vocab_size]
            hidden: New hidden state
            stack: New stack state
        """
        batch_size = inp.size(0) if inp.dim() > 0 else 1

        # Embed input
        embedded = self.embedding(inp.view(-1))  # [batch, embed_dim]
        embedded = embedded.unsqueeze(0)  # [1, batch, embed_dim]

        if self.has_stack:
            # Stack operations
            hidden_for_stack = hidden[-1]  # [batch, hidden_size]
            stack_ctrl = F.softmax(self.stack_controls(hidden_for_stack), dim=-1)
            stack_inp = torch.tanh(self.stack_input(hidden_for_stack))

            # Update stack
            stack = self._stack_operation(stack_inp, stack, stack_ctrl)

            # Concatenate with stack top
            stack_top = stack[:, 0, :].unsqueeze(0)  # [1, batch, stack_width]
            embedded = torch.cat([embedded, stack_top], dim=-1)

        # GRU step
        output, hidden = self.gru(embedded, hidden)

        # Output logits
        logits = self.output(output.squeeze(0))

        return logits, hidden, stack

    def _stack_operation(self, push_val, prev_stack, controls):
        """Differentiable stack operations."""
        batch_size = prev_stack.size(0)
        push, pop, noop = controls[:, 0:1, None], controls[:, 1:2, None], controls[:, 2:3, None]

        # Create shifted versions
        zeros = torch.zeros(batch_size, 1, self.stack_width, device=prev_stack.device)
        stack_down = torch.cat([prev_stack[:, 1:], zeros], dim=1)
        stack_up = torch.cat([push_val.unsqueeze(1), prev_stack[:, :-1]], dim=1)

        # Weighted combination
        new_stack = push * stack_up + pop * stack_down + noop * prev_stack
        return new_stack

    def init_hidden(self, batch_size=1):
        """Initialize hidden state."""
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if self.use_cuda:
            h = h.cuda()
        return h

    def init_stack(self, batch_size=1):
        """Initialize stack."""
        if not self.has_stack:
            return None
        s = torch.zeros(batch_size, self.stack_depth, self.stack_width)
        if self.use_cuda:
            s = s.cuda()
        return s

    def generate(self, data, start_token='<', end_token='>', max_len=100,
                 temperature=1.0, batch_size=1):
        """
        Generate SMILES strings.

        Args:
            data: GeneratorData with vocabulary
            start_token: Start token
            end_token: End token
            max_len: Maximum length
            temperature: Sampling temperature
            batch_size: Number of molecules to generate

        Returns:
            List of generated SMILES
        """
        self.eval()

        hidden = self.init_hidden(batch_size)
        stack = self.init_stack(batch_size)

        # Start tokens
        inp = data.char_tensor(start_token).expand(batch_size)
        if self.use_cuda:
            inp = inp.cuda()

        generated = [[start_token] for _ in range(batch_size)]
        finished = [False] * batch_size

        with torch.no_grad():
            for _ in range(max_len):
                logits, hidden, stack = self(inp, hidden, stack)

                # Temperature-scaled sampling
                probs = F.softmax(logits / temperature, dim=-1)
                chars = torch.multinomial(probs, 1).squeeze(-1)

                for i in range(batch_size):
                    if not finished[i]:
                        char = data.all_characters[chars[i].item()]
                        generated[i].append(char)
                        if char == end_token:
                            finished[i] = True

                if all(finished):
                    break

                inp = chars

        return [''.join(seq) for seq in generated]

    def sample_with_log_probs(self, data, start_token='<', end_token='>',
                               max_len=100, temperature=1.0):
        """
        Sample a molecule and return log probabilities for RL.

        Returns:
            smiles: Generated SMILES
            log_probs: List of log probabilities
            rewards: Placeholder for rewards
        """
        hidden = self.init_hidden(1)
        stack = self.init_stack(1)

        inp = data.char_tensor(start_token)
        if self.use_cuda:
            inp = inp.cuda()

        generated = [start_token]
        log_probs = []

        for _ in range(max_len):
            logits, hidden, stack = self(inp, hidden, stack)
            probs = F.softmax(logits / temperature, dim=-1)

            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            log_probs.append(log_prob)

            char = data.all_characters[action.item()]
            generated.append(char)

            if char == end_token:
                break

            inp = data.char_tensor(char)
            if self.use_cuda:
                inp = inp.cuda()

        smiles = ''.join(generated)
        return smiles, log_probs

    def train_step(self, inp, target):
        """
        Supervised training step.

        Args:
            inp: Input sequence [seq_len]
            target: Target sequence [seq_len]

        Returns:
            Loss value
        """
        self.train()

        hidden = self.init_hidden(1)
        stack = self.init_stack(1)

        losses = []
        for i in range(len(inp)):
            logits, hidden, stack = self(inp[i:i+1], hidden, stack)
            loss = self.criterion(logits, target[i:i+1])
            losses.append(loss)

        return torch.stack(losses).mean()

    def load_model(self, path):
        """Load pretrained weights."""
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict)
        if self.use_cuda:
            self.cuda()

    def save_model(self, path):
        """Save model weights."""
        torch.save(self.state_dict(), path)


class ConditionalMoleculeGenerator(MoleculeGenerator):
    """
    Conditional Molecule Generator that conditions on property embeddings.
    Useful for goal-directed molecule generation.
    """

    def __init__(self, vocab_size, property_dim=64, hidden_size=512,
                 embedding_dim=128, n_layers=2, has_stack=True,
                 stack_width=100, stack_depth=100, dropout=0.1, use_cuda=True):
        super(ConditionalMoleculeGenerator, self).__init__(
            vocab_size, hidden_size, embedding_dim, n_layers,
            has_stack, stack_width, stack_depth, dropout, use_cuda
        )

        self.property_dim = property_dim

        # Property encoder
        self.property_encoder = nn.Sequential(
            nn.Linear(property_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * n_layers)
        )

        # Modify GRU input size
        if has_stack:
            rnn_input_size = embedding_dim + stack_width + property_dim
        else:
            rnn_input_size = embedding_dim + property_dim

        self.gru = nn.GRU(
            rnn_input_size, hidden_size, n_layers,
            batch_first=False, dropout=dropout if n_layers > 1 else 0
        )

        if self.use_cuda:
            self.cuda()


class EncoderConditionedGenerator(ConditionalMoleculeGenerator):
    """Generator conditioned on the dual encoder output f ∈ R^{conditioning_dim}.

    The encoder output conditions generation in two ways:
    1. Hidden state initialization: h_0 = tanh(MLP(f))
    2. Per-step concatenation: input_t = [embed(char_t); stack_top; f]

    This makes the generator part of an end-to-end architecture where the
    dual encoder's learned chemistry directly drives molecule generation.
    """

    def __init__(self, vocab_size, conditioning_dim=128, hidden_size=512,
                 embedding_dim=128, n_layers=2, has_stack=True,
                 stack_width=100, stack_depth=200, dropout=0.1, use_cuda=True):
        # Initialize via ConditionalMoleculeGenerator with property_dim=conditioning_dim
        super(EncoderConditionedGenerator, self).__init__(
            vocab_size=vocab_size,
            property_dim=conditioning_dim,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            has_stack=has_stack,
            stack_width=stack_width,
            stack_depth=stack_depth,
            dropout=dropout,
            use_cuda=use_cuda,
        )
        self.conditioning_dim = conditioning_dim

    def init_hidden_with_property(self, property_vec, batch_size=1):
        """Initialize hidden state conditioned on property."""
        # Encode property
        encoded = self.property_encoder(property_vec)
        hidden = encoded.view(self.n_layers, batch_size, self.hidden_size)
        return hidden

    def forward(self, inp, hidden, stack=None, property_vec=None):
        """Forward pass with property conditioning."""
        batch_size = inp.size(0) if inp.dim() > 0 else 1

        embedded = self.embedding(inp.view(-1)).unsqueeze(0)

        if self.has_stack:
            hidden_for_stack = hidden[-1]
            stack_ctrl = F.softmax(self.stack_controls(hidden_for_stack), dim=-1)
            stack_inp = torch.tanh(self.stack_input(hidden_for_stack))
            stack = self._stack_operation(stack_inp, stack, stack_ctrl)
            stack_top = stack[:, 0, :].unsqueeze(0)
            embedded = torch.cat([embedded, stack_top], dim=-1)

        # Add property conditioning
        if property_vec is not None:
            prop = property_vec.unsqueeze(0).expand(1, batch_size, -1)
            embedded = torch.cat([embedded, prop], dim=-1)

        output, hidden = self.gru(embedded, hidden)
        logits = self.output(output.squeeze(0))

        return logits, hidden, stack

    def generate_with_property(self, data, property_vec, start_token='<',
                                end_token='>', max_len=100, temperature=1.0):
        """Generate conditioned on target properties."""
        self.eval()

        hidden = self.init_hidden_with_property(property_vec)
        stack = self.init_stack(1)

        inp = data.char_tensor(start_token)
        if self.use_cuda:
            inp = inp.cuda()
            property_vec = property_vec.cuda()

        generated = [start_token]

        with torch.no_grad():
            for _ in range(max_len):
                logits, hidden, stack = self(inp, hidden, stack, property_vec)
                probs = F.softmax(logits / temperature, dim=-1)
                char_idx = torch.multinomial(probs, 1).squeeze(-1)

                char = data.all_characters[char_idx.item()]
                generated.append(char)

                if char == end_token:
                    break

                inp = data.char_tensor(char)
                if self.use_cuda:
                    inp = inp.cuda()

        return ''.join(generated)
