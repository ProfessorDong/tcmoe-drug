"""
Policy Gradient Methods for Molecular Generation
Implements REINFORCE and variants for drug design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from rdkit import Chem


class PolicyGradient:
    """
    Policy Gradient trainer for molecule generation.
    Uses REINFORCE algorithm with baseline.
    """

    def __init__(self, generator, reward_fn, baseline_type='moving_avg',
                 gamma=0.97, entropy_coef=0.01, grad_clip=None):
        """
        Args:
            generator: MoleculeGenerator model
            reward_fn: Reward function that takes SMILES and returns reward
            baseline_type: Type of baseline ('moving_avg', 'learned', 'none')
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            grad_clip: Gradient clipping value
        """
        self.generator = generator
        self.reward_fn = reward_fn
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

        self.baseline_type = baseline_type
        self.baseline_value = 0.0
        self.baseline_alpha = 0.1  # Moving average coefficient

        if baseline_type == 'learned':
            self.baseline_net = nn.Sequential(
                nn.Linear(generator.hidden_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def train_step(self, data, n_batch=10, std_smiles=False):
        """
        Single training step using policy gradient.

        Args:
            data: GeneratorData with vocabulary
            n_batch: Number of trajectories to sample
            std_smiles: Standardize SMILES before reward

        Returns:
            mean_reward: Average reward
            mean_loss: Average policy loss
        """
        self.generator.train()
        self.generator.optimizer.zero_grad() if hasattr(self.generator, 'optimizer') else None

        total_reward = 0
        total_loss = 0
        total_entropy = 0

        for _ in range(n_batch):
            # Sample trajectory
            trajectory, log_probs, entropies = self._sample_trajectory(data)

            # Get reward
            smiles = trajectory[1:-1]  # Remove start/end tokens
            if std_smiles:
                smiles = self._standardize_smiles(smiles)

            reward = self.reward_fn(smiles)
            total_reward += reward

            # Compute discounted rewards
            rewards = self._compute_discounted_rewards(reward, len(log_probs))

            # Compute baseline
            baseline = self._get_baseline(rewards)

            # Policy gradient loss
            advantages = rewards - baseline
            policy_loss = 0
            for log_prob, advantage in zip(log_probs, advantages):
                policy_loss -= log_prob * advantage

            # Entropy bonus
            entropy_loss = -self.entropy_coef * sum(entropies)

            total_loss += policy_loss + entropy_loss
            total_entropy += sum(e.item() for e in entropies)

            # Update baseline
            self._update_baseline(np.mean(rewards))

        # Average and backprop
        total_loss = total_loss / n_batch
        total_loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), self.grad_clip
            )

        if hasattr(self.generator, 'optimizer'):
            self.generator.optimizer.step()

        return total_reward / n_batch, total_loss.item()

    def _sample_trajectory(self, data, max_len=100):
        """Sample a trajectory and return log probs."""
        hidden = self.generator.init_hidden(1)
        stack = self.generator.init_stack(1)

        inp = data.char_tensor('<')
        if self.generator.use_cuda:
            inp = inp.cuda()

        trajectory = '<'
        log_probs = []
        entropies = []

        for _ in range(max_len):
            logits, hidden, stack = self.generator(inp, hidden, stack)
            probs = F.softmax(logits, dim=-1)

            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())

            char = data.all_characters[action.item()]
            trajectory += char

            if char == '>':
                break

            inp = data.char_tensor(char)
            if self.generator.use_cuda:
                inp = inp.cuda()

        return trajectory, log_probs, entropies

    def _compute_discounted_rewards(self, final_reward, length):
        """Compute discounted rewards for each step."""
        rewards = []
        discounted = final_reward
        for _ in range(length):
            rewards.append(discounted)
            discounted *= self.gamma
        return list(reversed(rewards))

    def _get_baseline(self, rewards):
        """Get baseline value."""
        if self.baseline_type == 'none':
            return 0.0
        elif self.baseline_type == 'moving_avg':
            return self.baseline_value
        elif self.baseline_type == 'learned':
            # Not implemented for simplicity
            return self.baseline_value
        return 0.0

    def _update_baseline(self, reward):
        """Update moving average baseline."""
        self.baseline_value = (
            self.baseline_alpha * reward +
            (1 - self.baseline_alpha) * self.baseline_value
        )

    def _standardize_smiles(self, smiles):
        """Standardize SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                return Chem.MolToSmiles(mol)
        except:
            pass
        return smiles


class Reinforcement:
    """
    Reinforcement Learning wrapper for molecule generation.
    REINFORCE with moving-average baseline, entropy regularization,
    gradient clipping, and proper discounting (gamma^{T-t}).
    """

    def __init__(self, generator, predictor, get_reward,
                 entropy_coef=0.01, baseline_alpha=0.1):
        """
        Args:
            generator: MoleculeGenerator model
            predictor: Property predictor
            get_reward: Reward function(smiles, predictor, **kwargs) -> float
            entropy_coef: Entropy regularization coefficient (lambda_H)
            baseline_alpha: Moving-average baseline smoothing factor
        """
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward
        self.entropy_coef = entropy_coef
        self.baseline_alpha = baseline_alpha
        self.baseline = 0.0

    def policy_gradient(self, data, n_batch=10, gamma=0.97,
                        std_smiles=False, grad_clipping=1.0, **kwargs):
        """
        Policy gradient update with baseline, entropy, and clipping.

        Args:
            data: GeneratorData object
            n_batch: Batch size (molecules per step)
            gamma: Discount factor
            std_smiles: Standardize SMILES
            grad_clipping: Gradient norm clipping (default 1.0)

        Returns:
            mean_reward: Average reward
            mean_loss: Average loss
        """
        self.generator.train()

        if hasattr(self.generator, 'optimizer'):
            self.generator.optimizer.zero_grad()

        rl_loss = 0
        entropy_loss = 0
        total_reward = 0
        n_valid = 0

        for _ in range(n_batch):
            # Generate one molecule (allow R=0 for invalid SMILES)
            trajectory = self._generate(data)

            if std_smiles:
                trajectory = self._standardize(trajectory)

            smiles = trajectory[1:-1]
            reward = self.get_reward(smiles, self.predictor, **kwargs)
            total_reward += reward

            if reward != 0:
                n_valid += 1

            # Compute advantage: R - baseline
            advantage = reward - self.baseline

            # Replay trajectory to compute log-probs and entropy
            trajectory_tensor = data.char_tensor(trajectory)
            T = len(trajectory) - 1  # number of actions

            hidden = self.generator.init_hidden(1)
            stack = self.generator.init_stack(1)

            for i in range(T):
                inp = trajectory_tensor[i:i+1]
                if self.generator.use_cuda:
                    inp = inp.cuda()

                logits, hidden, stack = self.generator(inp, hidden, stack)
                log_probs = F.log_softmax(logits, dim=-1)
                probs = F.softmax(logits, dim=-1)

                target = trajectory_tensor[i + 1]

                # Correct discounting: gamma^{T-1-i} so last action
                # gets full credit (gamma^0) and first gets gamma^{T-1}
                discount = gamma ** (T - 1 - i)
                rl_loss -= log_probs[0, target] * discount * advantage

                # Entropy: H(pi) = -sum(p * log p)
                step_entropy = -(probs * log_probs).sum()
                entropy_loss -= step_entropy

        # Average over batch
        rl_loss = rl_loss / n_batch
        entropy_loss = entropy_loss / n_batch
        total_reward = total_reward / n_batch

        # Combined loss: policy gradient - entropy bonus
        combined_loss = rl_loss + self.entropy_coef * entropy_loss
        combined_loss.backward()

        # Gradient clipping
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), grad_clipping
            )

        if hasattr(self.generator, 'optimizer'):
            self.generator.optimizer.step()

        # Update moving-average baseline
        self.baseline = (self.baseline_alpha * total_reward +
                         (1 - self.baseline_alpha) * self.baseline)

        return total_reward, combined_loss.item()

    def _generate(self, data, max_len=100):
        """Generate single molecule."""
        hidden = self.generator.init_hidden(1)
        stack = self.generator.init_stack(1)

        inp = data.char_tensor('<')
        if self.generator.use_cuda:
            inp = inp.cuda()

        generated = '<'

        with torch.no_grad():
            for _ in range(max_len):
                logits, hidden, stack = self.generator(inp, hidden, stack)
                probs = F.softmax(logits, dim=-1)
                char_idx = torch.multinomial(probs, 1).squeeze(-1)

                char = data.all_characters[char_idx.item()]
                generated += char

                if char == '>':
                    break

                inp = data.char_tensor(char)
                if self.generator.use_cuda:
                    inp = inp.cuda()

        return generated

    def _standardize(self, trajectory):
        """Standardize SMILES in trajectory."""
        try:
            smiles = trajectory[1:-1]
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                return '<' + Chem.MolToSmiles(mol) + '>'
        except:
            pass
        return trajectory


class PPO:
    """
    Proximal Policy Optimization for molecule generation.
    More stable than vanilla policy gradient.
    """

    def __init__(self, generator, reward_fn, clip_epsilon=0.2,
                 value_coef=0.5, entropy_coef=0.01):
        """
        Args:
            generator: MoleculeGenerator model
            reward_fn: Reward function
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        self.generator = generator
        self.reward_fn = reward_fn
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(generator.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.optimizer = torch.optim.Adam(
            list(generator.parameters()) + list(self.value_net.parameters()),
            lr=1e-4
        )

    def train_step(self, data, n_batch=10, n_epochs=4):
        """PPO training step with multiple epochs."""
        # Collect trajectories
        trajectories = []
        for _ in range(n_batch):
            traj = self._collect_trajectory(data)
            trajectories.append(traj)

        # PPO epochs
        total_loss = 0
        for _ in range(n_epochs):
            loss = self._ppo_update(trajectories)
            total_loss += loss

        mean_reward = np.mean([t['reward'] for t in trajectories])
        return mean_reward, total_loss / n_epochs

    def _collect_trajectory(self, data):
        """Collect single trajectory."""
        self.generator.eval()

        hidden = self.generator.init_hidden(1)
        stack = self.generator.init_stack(1)

        inp = data.char_tensor('<')
        if self.generator.use_cuda:
            inp = inp.cuda()

        trajectory = '<'
        log_probs = []
        values = []

        with torch.no_grad():
            for _ in range(100):
                logits, hidden, stack = self.generator(inp, hidden, stack)
                probs = F.softmax(logits, dim=-1)
                dist = Categorical(probs)

                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.value_net(hidden[-1]).squeeze()

                log_probs.append(log_prob.item())
                values.append(value.item())

                char = data.all_characters[action.item()]
                trajectory += char

                if char == '>':
                    break

                inp = data.char_tensor(char)
                if self.generator.use_cuda:
                    inp = inp.cuda()

        reward = self.reward_fn(trajectory[1:-1])

        return {
            'trajectory': trajectory,
            'log_probs': log_probs,
            'values': values,
            'reward': reward
        }

    def _ppo_update(self, trajectories):
        """Single PPO update."""
        self.generator.train()

        total_loss = 0
        for traj in trajectories:
            # Recompute log probs
            # ... (simplified for brevity)
            pass

        return total_loss / len(trajectories)
