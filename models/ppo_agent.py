"""
PPO Agent for Coastal Emergency Warning System
Extracted and adapted from original Colab implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, Dict, List, Optional


class ActorCritic(nn.Module):
    """Actor-Critic network with attention mechanism"""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: List[int] = [256, 128, 64],
                 attention_heads: int = 4,
                 use_attention: bool = True):
        super(ActorCritic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_attention = use_attention

        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=prev_dim,
                num_heads=attention_heads,
                dropout=0.1,
                batch_first=True
            )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        """Forward pass through network"""
        # Extract features
        features = self.feature_extractor(state)

        # Apply attention if enabled
        if self.use_attention:
            # Reshape for attention: (batch, 1, features)
            features_attn = features.unsqueeze(1)
            features_attn, _ = self.attention(features_attn, features_attn, features_attn)
            features = features_attn.squeeze(1)

        # Actor output (logits)
        logits = self.actor(features)

        # Apply action masking if provided
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        # Critic output (value)
        value = self.critic(features)

        return logits, value

    def get_action(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None):
        """Select action using policy"""
        logits, value = self.forward(state, action_mask)

        # Create distribution and sample
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor,
                         action_masks: Optional[torch.Tensor] = None):
        """Evaluate actions for PPO update"""
        logits, values = self.forward(states, action_masks)

        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


class PPOAgent:
    """Proximal Policy Optimization Agent"""

    def __init__(self, state_dim: int, action_dim: int, config: Dict):
        """
        Initialize PPO agent

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary with training parameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create network
        self.policy = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.get('network', {}).get('hidden_dims', [256, 128, 64]),
            attention_heads=config.get('network', {}).get('attention_heads', 4),
            use_attention=config.get('network', {}).get('use_self_attention', True)
        ).to(self.device)

        # Optimizer
        lr = config.get('training', {}).get('learning_rate', 0.0003)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Training parameters
        self.gamma = config.get('training', {}).get('gamma', 0.99)
        self.gae_lambda = config.get('training', {}).get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('training', {}).get('clip_epsilon', 0.2)
        self.value_loss_coef = config.get('training', {}).get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('training', {}).get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('training', {}).get('gradient_clip', 0.5)
        self.update_epochs = config.get('training', {}).get('update_epochs', 4)

        # Memory
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'values': [],
            'dones': [],
            'action_masks': []
        }

    def select_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None):
        """Select action for given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        mask_tensor = None
        if action_mask is not None:
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor, mask_tensor)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done, action_mask=None):
        """Store transition in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['dones'].append(done)
        self.memory['action_masks'].append(action_mask if action_mask is not None else np.ones(self.action_dim, dtype=bool))

    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.memory['rewards'])
        values = np.array(self.memory['values'] + [next_value])
        dones = np.array(self.memory['dones'])

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values[:-1]

        return advantages, returns

    def update(self) -> Dict[str, float]:
        """Perform PPO update"""
        if len(self.memory['states']) == 0:
            return {}

        # Compute advantages
        next_value = 0.0  # Terminal state
        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = torch.LongTensor(self.memory['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        action_masks = torch.BoolTensor(np.array(self.memory['action_masks'])).to(self.device)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of updates
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.update_epochs):
            # Evaluate actions
            log_probs, values, entropy = self.policy.evaluate_actions(states, actions, action_masks)

            # Policy loss (clipped)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        # Clear memory
        self.clear_memory()

        # Return metrics
        return {
            'policy_loss': total_policy_loss / self.update_epochs,
            'value_loss': total_value_loss / self.update_epochs,
            'entropy': total_entropy / self.update_epochs
        }

    def clear_memory(self):
        """Clear memory buffer"""
        for key in self.memory:
            self.memory[key] = []

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])