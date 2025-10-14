"""
Base Option Class (Temporal Abstraction)
Option = (Initiation I, Policy π, Termination β)
Learned from GVF features (OaK FC-STOMP: Feature → Subtask → Option)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from meta.meta_optimizer import MetaOptimizerAdapter

class OptionPolicy(nn.Module):
    """Internal policy for an option"""

    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, state):
        """Get action logits"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        # Handle batching
        squeeze_output = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True

        logits = self.net(state)

        # Remove batch dimension if input was unbatched
        if squeeze_output:
            logits = logits.squeeze(0)

        return logits

    def select_action(self, state):
        """Select action from policy"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()
        return action


class Option:
    """
    Base Option class
    Learns internal policy to achieve a predictive subtask
    """

    def __init__(self, option_id, name, state_dim, action_dim,
                 max_length=10, hidden_size=64, lr=1e-3):
        self.option_id = option_id
        self.name = name
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        # Internal policy
        self.policy = OptionPolicy(state_dim, action_dim, hidden_size)

        # Execution tracking
        self.execution_count = 0
        self.success_count = 0
        self.total_duration = 0
        self.creation_step = 0

        # Windowed tracking for continual learning (Fix #2)
        self.recent_executions = deque(maxlen=50)  # Track last 50 executions (success/failure)

        # Value function for intra-option learning (optional)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.policy_optimizer = None
        self.value_optimizer = None
        self.policy_lr = lr
        self.value_lr = lr

    def initiation(self, state):
        """
        Initiation set I_o
        Default: can be initiated from any state
        Override in subclasses if needed
        """
        return True

    def termination(self, state):
        """
        Termination condition β_o(s)
        Must be overridden in subclasses
        Returns: (should_terminate, success)
        """
        raise NotImplementedError

    def compute_pseudo_reward(self, state):
        """
        Pseudo-reward for option's internal objective
        Must be overridden in subclasses
        """
        raise NotImplementedError

    def select_action(self, state):
        """Select action using option's internal policy"""
        return self.policy.select_action(state)

    def configure_optimizers(
        self,
        policy_lr=None,
        value_lr=None,
        policy_meta_config=None,
        value_meta_config=None,
    ):
        """Attach optimizers/meta-optimizers for the option."""
        self.policy_lr = policy_lr if policy_lr is not None else self.policy_lr
        self.value_lr = value_lr if value_lr is not None else self.value_lr
        self.policy_optimizer = MetaOptimizerAdapter(
            self.policy.parameters(),
            base_lr=self.policy_lr,
            meta_config=policy_meta_config,
        )
        self.value_optimizer = MetaOptimizerAdapter(
            self.value_net.parameters(),
            base_lr=self.value_lr,
            meta_config=value_meta_config,
        )

    def _ensure_optimizers(self):
        if self.policy_optimizer is None or self.value_optimizer is None:
            self.configure_optimizers()

    def update_policy(self, trajectory, gamma=0.99):
        """
        Update internal policy using actor-critic or policy gradient

        Args:
            trajectory: list of (s, a, r_pseudo, s') tuples
            gamma: discount factor
        """
        if len(trajectory) == 0:
            return 0.0

        total_loss = 0.0

        for t, (state, action, reward, next_state) in enumerate(trajectory):
            # Convert to tensors and ensure proper shapes
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            if isinstance(next_state, np.ndarray):
                next_state = torch.FloatTensor(next_state)

            # Ensure 1D state vectors
            if state.dim() == 0:
                state = state.unsqueeze(0)
            if next_state.dim() == 0:
                next_state = next_state.unsqueeze(0)

            # Value estimates (scalar outputs)
            v_s = self.value_net(state.unsqueeze(0) if state.dim() == 1 else state).squeeze()
            with torch.no_grad():
                v_s_next = self.value_net(next_state.unsqueeze(0) if next_state.dim() == 1 else next_state).squeeze()

            # Ensure scalar values
            if v_s.dim() > 0:
                v_s = v_s.squeeze()
            if v_s_next.dim() > 0:
                v_s_next = v_s_next.squeeze()

            # Ensure reward is scalar (convert if tensor)
            if isinstance(reward, torch.Tensor):
                if reward.dim() > 0:
                    reward = reward.item()

            # TD error (advantage) - scalar
            td_error = reward + gamma * v_s_next - v_s

            # Ensure td_error is truly scalar
            if isinstance(td_error, torch.Tensor) and td_error.dim() > 0:
                td_error = td_error.squeeze()

            # Policy gradient loss
            logits = self.policy(state)

            # Ensure logits has proper shape (action_dim,)
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            # Get log probability for selected action
            log_probs = F.log_softmax(logits, dim=-1)

            # Convert action to tensor if needed
            if isinstance(action, (int, np.integer)):
                action_idx = action
            else:
                action_idx = int(action)

            log_prob = log_probs[action_idx]
            policy_loss = -log_prob * td_error.detach()

            # Value loss
            value_loss = td_error.pow(2)

            # Combined loss - ensure scalar
            loss = policy_loss + 0.5 * value_loss

            # If somehow not scalar, take mean
            if loss.dim() > 0:
                loss = loss.mean()

            # Backward pass
            self._ensure_optimizers()
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)

            feature_vec = state.detach().cpu().numpy().reshape(-1)
            self.policy_optimizer.step(feature_vec, clip_range=(-10.0, 10.0))
            self.value_optimizer.step(feature_vec, clip_range=(-10.0, 10.0))

            total_loss += loss.item()

        return total_loss / len(trajectory)

    def execute(self, env, max_steps=None):
        """
        Execute option in environment until termination

        Args:
            env: environment
            max_steps: maximum steps (default: self.max_length)

        Returns:
            trajectory: list of (s, a, r_env, s', done)
            success: whether option achieved its goal
        """
        if max_steps is None:
            max_steps = self.max_length

        trajectory = []
        state = env.state
        steps = 0
        success = False

        while steps < max_steps:
            # Select action using option policy
            action = self.select_action(state)

            # Execute in environment
            next_state, reward, done, info = env.step(action)

            # Store transition
            trajectory.append((state, action, reward, next_state, done))

            # Check termination
            should_terminate, achieved = self.termination(next_state)

            if achieved:
                success = True

            if should_terminate or done:
                break

            state = next_state
            steps += 1

        # Update statistics
        self.execution_count += 1
        self.total_duration += len(trajectory)
        if success:
            self.success_count += 1

        # Windowed tracking for continual learning (Fix #2)
        self.recent_executions.append(1 if success else 0)

        return trajectory, success

    def get_statistics(self):
        """Get option execution statistics"""
        if self.execution_count == 0:
            return {
                'executions': 0,
                'success_rate': 0.0,
                'avg_duration_steps': 0.0
            }

        return {
            'executions': self.execution_count,
            'success_rate': self.success_count / self.execution_count,
            'avg_duration_steps': self.total_duration / self.execution_count
        }

    def get_recent_success_rate(self, window=None):
        """
        Get success rate over recent executions (Fix #2: continual learning)

        Args:
            window: Number of recent executions to consider (default: use deque maxlen)

        Returns:
            Success rate (0.0 to 1.0) or None if insufficient data
        """
        if len(self.recent_executions) == 0:
            return None

        if window is not None and window < len(self.recent_executions):
            recent_subset = list(self.recent_executions)[-window:]
            return sum(recent_subset) / len(recent_subset)

        return sum(self.recent_executions) / len(self.recent_executions)
