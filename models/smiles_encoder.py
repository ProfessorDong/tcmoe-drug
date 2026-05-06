"""
SMILES Sequence Encoder
Implements RNN-based encoding with optional stack augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class StackAugmentedRNN(nn.Module):
    """
    Stack-Augmented Recurrent Neural Network for SMILES generation.
    Based on https://arxiv.org/abs/1503.01007

    The stack provides external memory for learning context-free grammars,
    which is useful for generating valid SMILES with balanced brackets.
    """

    def __init__(self, input_size, hidden_size, output_size, layer_type='GRU',
                 n_layers=1, bidirectional=False, has_stack=True,
                 stack_width=100, stack_depth=100, dropout=0.0,
                 use_cuda=True):
        super(StackAugmentedRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layer_type = layer_type
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.has_stack = has_stack
        self.stack_width = stack_width
        self.stack_depth = stack_depth
        self.use_cuda = use_cuda and torch.cuda.is_available()

        # Embedding layer
        self.encoder = nn.Embedding(input_size, hidden_size)

        # Stack control layers (PUSH, POP, NO_OP)
        if has_stack:
            self.stack_controls = nn.Linear(
                hidden_size * self.num_directions, 3
            )
            self.stack_input = nn.Linear(
                hidden_size * self.num_directions, stack_width
            )
            rnn_input_size = hidden_size + stack_width
        else:
            rnn_input_size = hidden_size

        # RNN layer
        if layer_type == 'GRU':
            self.rnn = nn.GRU(
                rnn_input_size, hidden_size, n_layers,
                batch_first=False, bidirectional=bidirectional, dropout=dropout
            )
        elif layer_type == 'LSTM':
            self.rnn = nn.LSTM(
                rnn_input_size, hidden_size, n_layers,
                batch_first=False, bidirectional=bidirectional, dropout=dropout
            )
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        # Decoder
        self.decoder = nn.Linear(hidden_size * self.num_directions, output_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, hidden, stack=None):
        """
        Single step forward pass.

        Args:
            inp: Input character index [1] or [batch_size]
            hidden: Previous hidden state
            stack: Previous stack state (if has_stack)

        Returns:
            output: Logits for next character [1, output_size]
            hidden: New hidden state
            stack: New stack state
        """
        # Embed input
        embedded = self.encoder(inp.view(1, -1))  # [1, batch, hidden]

        if self.has_stack:
            # Get hidden for stack operations
            if self.layer_type == 'LSTM':
                hidden_for_stack = hidden[0]
            else:
                hidden_for_stack = hidden

            # Combine bidirectional hidden states if needed
            if self.bidirectional:
                hidden_combined = torch.cat(
                    [hidden_for_stack[-2], hidden_for_stack[-1]], dim=1
                )
            else:
                hidden_combined = hidden_for_stack[-1]

            # Stack operations
            stack_controls = F.softmax(self.stack_controls(hidden_combined), dim=1)
            stack_input = torch.tanh(self.stack_input(hidden_combined))

            # Update stack
            stack = self._stack_augmentation(
                stack_input.unsqueeze(1), stack, stack_controls
            )

            # Get stack top and concatenate with input
            stack_top = stack[:, 0, :].unsqueeze(0)
            embedded = torch.cat([embedded, stack_top], dim=2)

        # RNN step
        output, hidden = self.rnn(embedded, hidden)

        # Decode
        output = self.decoder(output.view(1, -1))

        return output, hidden, stack

    def _stack_augmentation(self, stack_input, prev_stack, controls):
        """
        Differentiable stack operations.

        Args:
            stack_input: Value to push [batch, 1, stack_width]
            prev_stack: Previous stack [batch, stack_depth, stack_width]
            controls: [push, pop, no_op] probabilities [batch, 3]

        Returns:
            New stack state
        """
        batch_size = prev_stack.size(0)

        # Reshape controls for broadcasting
        controls = controls.view(-1, 3, 1, 1)
        push, pop, no_op = controls[:, 0], controls[:, 1], controls[:, 2]

        # Stack operations
        zeros_bottom = torch.zeros(batch_size, 1, self.stack_width)
        if self.use_cuda:
            zeros_bottom = zeros_bottom.cuda()

        # Pop: shift down (lose bottom)
        stack_down = torch.cat([prev_stack[:, 1:], zeros_bottom], dim=1)

        # Push: shift up (new on top)
        stack_up = torch.cat([stack_input, prev_stack[:, :-1]], dim=1)

        # Combine with soft attention
        new_stack = no_op * prev_stack + push * stack_up + pop * stack_down

        return new_stack

    def init_hidden(self, batch_size=1):
        """Initialize hidden state."""
        h = torch.zeros(self.n_layers * self.num_directions, batch_size, self.hidden_size)
        if self.use_cuda:
            h = h.cuda()

        if self.layer_type == 'LSTM':
            c = torch.zeros_like(h)
            return (h, c)
        return h

    def init_stack(self, batch_size=1):
        """Initialize stack state."""
        if not self.has_stack:
            return None
        stack = torch.zeros(batch_size, self.stack_depth, self.stack_width)
        if self.use_cuda:
            stack = stack.cuda()
        return stack


class SMILESEncoder(nn.Module):
    """
    Bidirectional SMILES Encoder with attention mechanism.
    Encodes SMILES strings to fixed-size representations.
    """

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 output_dim=128, n_layers=2, bidirectional=True,
                 dropout=0.1, attention=True):
        super(SMILESEncoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.attention = attention

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional GRU
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0
        )

        # Attention mechanism
        if attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_dim * self.num_directions, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Store attention weights for interpretability
        self.attention_weights = None

    def forward(self, x, lengths=None, return_attention=False):
        """
        Encode SMILES sequences.

        Args:
            x: Tokenized SMILES [batch, seq_len]
            lengths: Sequence lengths [batch]
            return_attention: Whether to return attention weights

        Returns:
            Encoded representation [batch, output_dim]
        """
        batch_size, seq_len = x.size()

        # Embed
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]

        # Pack if lengths provided
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # GRU forward
        outputs, hidden = self.gru(embedded)

        # Unpack
        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=seq_len
            )

        # Apply attention or use last hidden state
        if self.attention:
            # Compute attention weights
            att_weights = self.attention_layer(outputs).squeeze(-1)  # [batch, seq_len]

            # Mask padded positions
            if lengths is not None:
                mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
                mask = mask >= lengths.to(x.device).unsqueeze(1)
                att_weights = att_weights.masked_fill(mask, float('-inf'))

            att_weights = F.softmax(att_weights, dim=1)
            self.attention_weights = att_weights

            # Weighted sum
            context = torch.bmm(att_weights.unsqueeze(1), outputs).squeeze(1)
        else:
            # Use final hidden state
            if self.bidirectional:
                context = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                context = hidden[-1]

        # Project to output
        output = self.output_proj(context)

        if return_attention and self.attention:
            return output, self.attention_weights

        return output

    def forward_with_sequence(self, x, lengths=None):
        """Forward pass returning both pooled output AND pre-pooling
        token embeddings for pre-pooling cross-attention.

        Args:
            x: Token indices [batch, seq_len]
            lengths: Actual lengths [batch]

        Returns:
            output: Pooled representation [batch, output_dim]
            sequence_embeddings: BiGRU outputs [batch, seq_len, hidden*2]
                (before attention pooling)
        """
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)

        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        outputs, hidden = self.gru(embedded)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True, total_length=seq_len
            )

        # outputs: [batch, seq_len, hidden_dim * num_directions]
        sequence_embeddings = outputs  # save before pooling

        # Pool (same as forward)
        if self.attention:
            att_weights = self.attention_layer(outputs).squeeze(-1)
            if lengths is not None:
                mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
                mask = mask >= lengths.to(x.device).unsqueeze(1)
                att_weights = att_weights.masked_fill(mask, float('-inf'))
            att_weights = F.softmax(att_weights, dim=1)
            context = torch.bmm(att_weights.unsqueeze(1), outputs).squeeze(1)
        else:
            if self.bidirectional:
                context = torch.cat([hidden[-2], hidden[-1]], dim=1)
            else:
                context = hidden[-1]

        output = self.output_proj(context)
        return output, sequence_embeddings

    def get_sequence_embeddings(self, x, lengths=None):
        """Get embeddings at each position for visualization."""
        embedded = self.embedding(x)

        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        outputs, _ = self.gru(embedded)

        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True
            )

        return outputs


class SMILESGenerator(nn.Module):
    """
    SMILES Generator using Stack-Augmented RNN.
    Can generate molecules autoregressively or with teacher forcing.
    """

    def __init__(self, vocab_size, hidden_size=512, n_layers=1,
                 has_stack=True, stack_width=100, stack_depth=100,
                 dropout=0.0, use_cuda=True):
        super(SMILESGenerator, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.rnn = StackAugmentedRNN(
            input_size=vocab_size,
            hidden_size=hidden_size,
            output_size=vocab_size,
            layer_type='GRU',
            n_layers=n_layers,
            has_stack=has_stack,
            stack_width=stack_width,
            stack_depth=stack_depth,
            dropout=dropout,
            use_cuda=use_cuda
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, hidden=None, stack=None):
        """Forward pass for single character."""
        return self.rnn(x, hidden, stack)

    def train_step(self, inp, target):
        """
        Training step with teacher forcing.

        Args:
            inp: Input sequence [seq_len]
            target: Target sequence [seq_len]

        Returns:
            Loss value
        """
        hidden = self.rnn.init_hidden()
        stack = self.rnn.init_stack()

        loss = 0
        for i in range(len(inp)):
            output, hidden, stack = self.rnn(inp[i], hidden, stack)
            loss += self.criterion(output, target[i].unsqueeze(0))

        return loss / len(inp)

    def generate(self, data, start_token='<', end_token='>', max_len=100,
                 temperature=1.0):
        """
        Generate a new SMILES string.

        Args:
            data: GeneratorData object with vocabulary
            start_token: Start of sequence token
            end_token: End of sequence token
            max_len: Maximum generation length
            temperature: Sampling temperature

        Returns:
            Generated SMILES string
        """
        hidden = self.rnn.init_hidden()
        stack = self.rnn.init_stack()

        inp = data.char_tensor(start_token)
        generated = start_token

        for _ in range(max_len):
            output, hidden, stack = self.rnn(inp, hidden, stack)

            # Temperature-scaled sampling
            probs = F.softmax(output / temperature, dim=1)
            top_i = torch.multinomial(probs.view(-1), 1)[0].item()

            char = data.all_characters[top_i]
            generated += char

            if char == end_token:
                break

            inp = data.char_tensor(char)

        return generated

    def load_model(self, path):
        """Load pretrained weights."""
        self.load_state_dict(torch.load(path, map_location='cpu'))

    def save_model(self, path):
        """Save model weights."""
        torch.save(self.state_dict(), path)
