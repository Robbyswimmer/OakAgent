"""
Primitive Q-Function (Double Q-Learning)
Value function for primitive actions
Trained on both real and simulated experience (Dyna principle)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.idbd import TorchIDBD

class QNetwork(nn.Module):
    """Q-network for primitive actions"""

    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) or (state_dim,)

        Returns:
            q_values: (batch, action_dim) or (action_dim,)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            return self.net(state).squeeze(0)

        return self.net(state)


class DoubleQNetwork:
    """
    Double Q-Learning with target network
    Learns Q(s, a) from real and simulated transitions
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.99,
        lr=1e-3,
        target_sync_freq=500,
        meta_config=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_sync_freq = target_sync_freq

        self.meta_config = meta_config or {}
        self.use_meta = meta_config is not None and not self.meta_config.get(
            "disabled", False
        )
        self.base_lr = lr

        # Q-network and target network
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # Optimizer / meta-learner
        self.meta_updater = None
        self.optimizer = None
        if self.use_meta:
            self.meta_updater = TorchIDBD(
                self.q_net.parameters(),
                mu=self.meta_config.get("mu", 1e-3),
                init_log_alpha=self.meta_config.get("init_log_alpha", np.log(lr)),
                meta_type=self.meta_config.get("type", "idbd"),
                min_alpha=self.meta_config.get("min_alpha", 1e-6),
                max_alpha=self.meta_config.get("max_alpha", 1.0),
            )
        else:
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Tracking
        self.update_count = 0
        self.td_errors = []

    def predict(self, state):
        """Get Q-values for state"""
        with torch.no_grad():
            return self.q_net(state).cpu().numpy()

    def select_action(self, state, epsilon=0.0):
        """Epsilon-greedy action selection"""
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)

        q_values = self.predict(state)
        return np.argmax(q_values)

    def update_td(self, state, action, reward, next_state, done):
        """
        TD update: Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q_target(s',a') - Q(s,a)]

        Args:
            state: (batch, state_dim)
            action: (batch,)
            reward: (batch,)
            next_state: (batch, state_dim)
            done: (batch,)

        Returns:
            loss value
        """
        # Convert to tensors
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.LongTensor(action)
        if isinstance(reward, np.ndarray):
            reward = torch.FloatTensor(reward)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
        if isinstance(done, np.ndarray):
            done = torch.FloatTensor(done)

        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0) if action.dim() == 0 else action
            reward = reward.unsqueeze(0) if reward.dim() == 0 else reward
            next_state = next_state.unsqueeze(0)
            done = done.unsqueeze(0) if done.dim() == 0 else done

        # Ensure action, reward, done are 1D (batch,) not (batch, 1)
        if action.dim() == 2:
            action = action.squeeze(1)
        if reward.dim() == 2:
            reward = reward.squeeze(1)
        if done.dim() == 2:
            done = done.squeeze(1)

        # Current Q-values
        q_values = self.q_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double Q-learning)
        with torch.no_grad():
            # Select action using online network
            next_q_values_online = self.q_net(next_state)
            next_actions = next_q_values_online.argmax(dim=1)

            # Evaluate action using target network
            next_q_values_target = self.target_net(next_state)
            next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # TD target
            target_q_value = reward + self.gamma * next_q_value * (1 - done)

        # Loss
        loss = F.mse_loss(q_value, target_q_value)

        # Backward pass with meta step-size adaptation if enabled
        if self.use_meta and self.meta_updater is not None:
            self.q_net.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [param.grad for param in self.q_net.parameters()]
            step_sizes = self.meta_updater.update_step_sizes(gradients, state)
            self.meta_updater.apply_updates(self.q_net.parameters(), step_sizes)
            self.q_net.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Track TD error
        td_error = (target_q_value - q_value).abs().mean().item()
        self.td_errors.append(td_error)

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_sync_freq == 0:
            self.sync_target()

        return loss.item()

    def sync_target(self):
        """Synchronize target network with Q-network"""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def get_average_td_error(self, n=100):
        """Get average TD error over last n updates"""
        if len(self.td_errors) == 0:
            return 0.0
        recent_errors = self.td_errors[-n:]
        return np.mean(recent_errors)
