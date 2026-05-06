"""
Theranostics Drug Discovery Framework - Main Example
=====================================================

This script demonstrates the complete workflow:
1. Dual representation (Graph + SMILES) encoding
2. Meta-learning (MAML) for rapid adaptation
3. Reinforcement learning for molecule generation
4. Attention visualization and interpretability

Based on run_example_1.ipynb and run_example_2.ipynb from the original codebase.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm

# RDKit imports
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP

# Framework imports
from models.dual_encoder import DualEncoder
from models.property_predictor import PropertyPredictor, OneShotPredictor
from models.molecule_generator import MoleculeGenerator
from data.loader import GeneratorData, load_smiles_from_file
from data.featurizer import SMILESTokenizer, GraphFeaturizer
from data.dataset import DualDataset, collate_dual
from meta_learning.maml import MAML, MAMLTrainer
from meta_learning.support_learning import SupportLearner, EpisodeSampler
from reinforcement.policy_gradient import Reinforcement
from reinforcement.reward_functions import LogPReward, MultiObjectiveReward
from interpretability.attention_viz import AttentionVisualizer
from utils.chemistry import canonical_smiles, compute_properties
from utils.training import Trainer, EarlyStopping


# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    data_path = '../ReLeaSE-master/data/chembl_22_clean_1576904_sorted_std_final.smi'
    checkpoint_path = '../ReLeaSE-master/checkpoints/'
    save_path = './checkpoints/'

    # Model parameters
    hidden_size = 512
    stack_width = 100
    stack_depth = 200
    embedding_dim = 128
    graph_hidden_dims = [128, 256, 128]

    # Training parameters
    batch_size = 32
    learning_rate = 1e-4
    n_epochs = 100
    n_inner_steps = 5  # MAML inner loop steps

    # Generation parameters
    n_to_generate = 200
    temperature = 1.0

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Tokens
    tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']


config = Config()


# ============================================================================
# Helper Functions
# ============================================================================

def plot_distribution(predictions, title='Distribution'):
    """Plot distribution of predictions."""
    predictions = np.array(predictions)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(predictions, bins=50, density=True, alpha=0.7, color='blue')
    ax.axvline(x=0.0, color='red', linestyle='--', label='Lower bound')
    ax.axvline(x=5.0, color='red', linestyle='--', label='Upper bound')

    ax.set_xlabel('Predicted Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig


def estimate_and_update(generator, predictor, n_to_generate, gen_data):
    """Generate molecules and estimate properties."""
    generated = []

    for _ in tqdm(range(n_to_generate), desc='Generating molecules...'):
        smiles = generator.generate(
            gen_data,
            start_token='<',
            end_token='>',
            max_len=120,
            temperature=config.temperature
        )[0]
        generated.append(smiles[1:-1])  # Remove start/end tokens

    # Canonicalize and filter
    sanitized = canonical_smiles(generated, sanitize=False)
    unique_smiles = list(set([s for s in sanitized if s is not None]))

    # Predict properties
    predictions = []
    valid_smiles = []

    for smiles in unique_smiles:
        props = compute_properties(smiles, ['logp'])
        if props and props.get('logp') is not None:
            predictions.append(props['logp'])
            valid_smiles.append(smiles)

    return valid_smiles, predictions


# ============================================================================
# 1. Data Loading
# ============================================================================

def setup_data():
    """Load and prepare data."""
    print("=" * 60)
    print("1. Loading Data")
    print("=" * 60)

    # Generator data
    gen_data = GeneratorData(
        training_data_path=config.data_path,
        tokens=config.tokens,
        delimiter='\t',
        cols_to_read=[0],
        keep_header=True
    )

    print(f"Loaded {gen_data.file_len} SMILES")
    print(f"Vocabulary size: {gen_data.n_characters}")

    # Tokenizer
    tokenizer = SMILESTokenizer(tokens=config.tokens)

    return gen_data, tokenizer


# ============================================================================
# 2. Model Setup
# ============================================================================

def setup_models(gen_data, tokenizer):
    """Initialize all models."""
    print("\n" + "=" * 60)
    print("2. Setting Up Models")
    print("=" * 60)

    # 2.1 Dual Encoder
    print("\n2.1 Initializing Dual Encoder...")
    dual_encoder = DualEncoder(
        node_features=75,
        edge_features=6,
        vocab_size=tokenizer.vocab_size,
        graph_hidden_dims=config.graph_hidden_dims,
        smiles_hidden_dim=256,
        output_dim=config.embedding_dim,
        graph_type='gat',
        num_heads=4,
        dropout=0.1,
        fusion_type='attention'
    ).to(config.device)

    print(f"  - Dual Encoder parameters: {sum(p.numel() for p in dual_encoder.parameters()):,}")

    # 2.2 Property Predictor
    print("\n2.2 Initializing Property Predictor...")
    task_configs = {
        'logp_class_1': {'type': 'classification', 'num_classes': 2},
        'logp_class_2': {'type': 'classification', 'num_classes': 2},
        'logp_class_3': {'type': 'classification', 'num_classes': 2},
    }

    predictor = PropertyPredictor(
        input_dim=config.embedding_dim,
        task_configs=task_configs,
        hidden_dims=[256, 128],
        dropout=0.1,
        use_bayesian=False
    ).to(config.device)

    print(f"  - Predictor parameters: {sum(p.numel() for p in predictor.parameters()):,}")

    # 2.3 MAML-wrapped Predictor
    print("\n2.3 Wrapping with MAML...")
    maml_model = MAML(
        predictor,
        inner_lr=0.01,
        first_order=True  # Faster approximation
    )

    # 2.4 Molecule Generator
    print("\n2.4 Initializing Molecule Generator...")
    generator = MoleculeGenerator(
        vocab_size=gen_data.n_characters,
        hidden_size=config.hidden_size,
        n_layers=1,
        has_stack=True,
        stack_width=config.stack_width,
        stack_depth=config.stack_depth,
        dropout=0.0,
        use_cuda=(config.device == 'cuda')
    )

    # Load pretrained generator if available
    pretrained_path = os.path.join(config.checkpoint_path, 'generator/checkpoint_biggest_rnn')
    if os.path.exists(pretrained_path):
        print(f"  - Loading pretrained generator from {pretrained_path}")
        generator.load_model(pretrained_path)
    else:
        print("  - No pretrained generator found, using random initialization")

    print(f"  - Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")

    return dual_encoder, predictor, maml_model, generator


# ============================================================================
# 3. One-Shot Predictor Setup
# ============================================================================

def setup_oneshot_predictor(dual_encoder):
    """Setup one-shot predictor using dual encoder."""
    print("\n" + "=" * 60)
    print("3. Setting Up One-Shot Predictor")
    print("=" * 60)

    oneshot = OneShotPredictor(
        encoder=dual_encoder,
        embedding_dim=config.embedding_dim,
        n_support=20
    ).to(config.device)

    print(f"  - One-shot predictor ready")

    return oneshot


# ============================================================================
# 4. MAML Training
# ============================================================================

def train_maml(maml_model, train_data, val_data=None, n_epochs=10):
    """Train model with MAML."""
    print("\n" + "=" * 60)
    print("4. MAML Training")
    print("=" * 60)

    trainer = MAMLTrainer(
        maml_model,
        meta_lr=config.learning_rate,
        inner_lr=0.01,
        inner_steps=config.n_inner_steps
    )

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        # Training
        train_loss = trainer.train_epoch(train_data, n_tasks_per_batch=4)
        history['train_loss'].append(train_loss)

        print(f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {train_loss:.4f}")

        # Validation
        if val_data is not None:
            val_results = trainer.evaluate(val_data, n_tasks=50)
            history['val_loss'].append(val_results['post_adapt_loss'])
            print(f"  Val Loss: {val_results['post_adapt_loss']:.4f}")
            print(f"  Improvement: {val_results['improvement']:.4f}")

    return history


# ============================================================================
# 5. Reinforcement Learning
# ============================================================================

def train_rl(generator, predictor, gen_data, n_iterations=100):
    """Train generator with RL."""
    print("\n" + "=" * 60)
    print("5. Reinforcement Learning Training")
    print("=" * 60)

    # Setup reward function
    logp_reward = LogPReward(target_range=(1.0, 4.0), use_rdkit=True)

    # Define reward wrapper
    def get_reward(smiles, pred, invalid_reward=0.0):
        return logp_reward(smiles)

    # Setup RL
    rl = Reinforcement(generator, predictor, get_reward)

    # Setup optimizer
    optimizer = Adam(generator.parameters(), lr=1e-3)
    generator.optimizer = optimizer

    # Training loop
    rewards = []
    losses = []

    for i in range(n_iterations):
        # Policy gradient steps
        for _ in range(10):
            reward, loss = rl.policy_gradient(gen_data, n_batch=10, gamma=0.97)
            rewards.append(reward)
            losses.append(loss)

        # Logging
        if (i + 1) % 10 == 0:
            mean_reward = np.mean(rewards[-100:])
            mean_loss = np.mean(losses[-100:])
            print(f"Iteration {i + 1}/{n_iterations} - "
                  f"Reward: {mean_reward:.4f}, Loss: {mean_loss:.4f}")

    return rewards, losses


# ============================================================================
# 6. Generation and Evaluation
# ============================================================================

def generate_and_evaluate(generator, gen_data, n_molecules=1000):
    """Generate molecules and evaluate."""
    print("\n" + "=" * 60)
    print("6. Generation and Evaluation")
    print("=" * 60)

    # Generate molecules
    generated_smiles = []
    for _ in tqdm(range(n_molecules), desc='Generating'):
        smiles = generator.generate(
            gen_data,
            start_token='<',
            end_token='>',
            max_len=120,
            temperature=config.temperature
        )[0]
        generated_smiles.append(smiles[1:-1])

    # Evaluate
    valid_smiles = [s for s in generated_smiles if Chem.MolFromSmiles(s) is not None]
    unique_smiles = list(set(valid_smiles))

    print(f"\nGeneration Statistics:")
    print(f"  - Total generated: {len(generated_smiles)}")
    print(f"  - Valid: {len(valid_smiles)} ({100 * len(valid_smiles) / len(generated_smiles):.1f}%)")
    print(f"  - Unique: {len(unique_smiles)} ({100 * len(unique_smiles) / len(valid_smiles):.1f}%)")

    # Compute properties
    logp_values = []
    for smiles in unique_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            logp_values.append(MolLogP(mol))

    print(f"\nLogP Statistics:")
    print(f"  - Mean: {np.mean(logp_values):.2f}")
    print(f"  - Std: {np.std(logp_values):.2f}")
    print(f"  - In range [1, 4]: {100 * np.mean((np.array(logp_values) >= 1) & (np.array(logp_values) <= 4)):.1f}%")

    return unique_smiles, logp_values


# ============================================================================
# 7. Visualization
# ============================================================================

def visualize_results(logp_values, save_path=None):
    """Visualize generation results."""
    print("\n" + "=" * 60)
    print("7. Visualization")
    print("=" * 60)

    fig = plot_distribution(logp_values, 'LogP Distribution of Generated Molecules')

    if save_path:
        fig.savefig(save_path)
        print(f"  - Saved to {save_path}")

    plt.show()

    return fig


def visualize_attention(dual_encoder, smiles, tokenizer, save_path=None):
    """Visualize attention weights."""
    print("\n" + "=" * 60)
    print("8. Attention Visualization")
    print("=" * 60)

    visualizer = AttentionVisualizer(dual_encoder, device=config.device)

    # Prepare input
    tokens = tokenizer.batch_encode([smiles], max_length=100)
    tokens = tokens[0].to(config.device)
    lengths = torch.tensor([len(smiles)])

    # Get attention
    attention = visualizer.get_smiles_attention(tokens.unsqueeze(0), lengths)

    print(f"  - SMILES: {smiles}")
    print(f"  - Top attention positions: {np.argsort(attention)[-5:][::-1]}")

    return attention


# ============================================================================
# Main
# ============================================================================

def main():
    """Main workflow."""
    print("=" * 60)
    print("Theranostics Drug Discovery Framework")
    print("Dual Graph + SMILES | Meta-RL (MAML) | Interpretability")
    print("=" * 60)

    # 1. Setup data
    gen_data, tokenizer = setup_data()

    # 2. Setup models
    dual_encoder, predictor, maml_model, generator = setup_models(gen_data, tokenizer)

    # 3. Setup one-shot predictor
    oneshot = setup_oneshot_predictor(dual_encoder)

    # 4. Demonstrate generation (skip MAML training for demo)
    print("\n" + "=" * 60)
    print("Demonstration: Molecule Generation")
    print("=" * 60)

    # Generate initial molecules
    smiles, predictions = estimate_and_update(generator, None, 100, gen_data)

    print(f"\nGenerated {len(smiles)} valid unique molecules")
    if predictions:
        print(f"LogP range: [{min(predictions):.2f}, {max(predictions):.2f}]")

    # 5. Show sample molecules
    print("\nSample generated molecules:")
    for i, (smi, pred) in enumerate(zip(smiles[:5], predictions[:5])):
        print(f"  {i + 1}. {smi} (LogP = {pred:.2f})")

    print("\n" + "=" * 60)
    print("Framework demonstration complete!")
    print("=" * 60)
    print("\nTo run full training:")
    print("  1. Load your dataset")
    print("  2. Call train_maml() for meta-learning")
    print("  3. Call train_rl() for RL fine-tuning")
    print("  4. Call generate_and_evaluate() for generation")


if __name__ == '__main__':
    main()
