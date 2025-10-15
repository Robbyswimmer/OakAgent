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


def _resolve_device(device):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, torch.device):
        return device
    return torch.device(device)

class QNetwork(nn.Module):
    """Q-network for primitive actions with optional state encoder"""

    def __init__(self, state_dim, action_dim, hidden_size=128, state_encoder=None, device=None):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = _resolve_device(device)
        if isinstance(state_encoder, nn.Module):
            self._modules.pop('_state_encoder', None)
            object.__setattr__(self, '_state_encoder', state_encoder)
        else:
            self._state_encoder = state_encoder

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

        self.to(self.device)

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) or (state_dim,)

        Returns:
            q_values: (batch, action_dim) or (action_dim,)
        """
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        if self._state_encoder is not None:
            with torch.no_grad():
                state = self._state_encoder.encode_tensor(state)

        output = self.net(state)
        if output.dim() == 2 and output.size(0) == 1:
            return output.squeeze(0)
        return output


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
        state_encoder=None,
        latent_dim=None,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_sync_freq = target_sync_freq
        self._state_encoder = state_encoder
        self.device = _resolve_device(device)

        if latent_dim is None:
            latent_dim = state_dim
        self.latent_dim = latent_dim

        self.meta_config = meta_config or {}
        self.use_meta = meta_config is not None and not self.meta_config.get(
            "disabled", False
        )
        self.base_lr = lr

        # Q-network and target network
        self.q_net = QNetwork(latent_dim, action_dim, state_encoder=state_encoder, device=self.device)
        self.target_net = QNetwork(latent_dim, action_dim, state_encoder=state_encoder, device=self.device)
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

    def resize_action_dim(self, new_action_dim):
        """Adjust networks when the action space size changes."""
        if int(new_action_dim) == self.action_dim:
            return

        new_action_dim = int(new_action_dim)
        self.action_dim = new_action_dim

        def _resize_network(net: QNetwork, action_dim: int):
            last_layer = net.net[-1]
            if last_layer.out_features == action_dim:
                return
            new_layer = nn.Linear(last_layer.in_features, action_dim, device=self.device)
            with torch.no_grad():
                overlap = min(last_layer.out_features, action_dim)
                if overlap > 0:
                    new_layer.weight[:overlap] = last_layer.weight[:overlap]
                    if last_layer.bias is not None and new_layer.bias is not None:
                        new_layer.bias[:overlap] = last_layer.bias[:overlap]
            net.net[-1] = new_layer
            net.action_dim = action_dim

        _resize_network(self.q_net, new_action_dim)
        _resize_network(self.target_net, new_action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        if self.use_meta:
            self.meta_updater = TorchIDBD(
                self.q_net.parameters(),
                mu=self.meta_config.get("mu", 1e-3),
                init_log_alpha=self.meta_config.get(
                    "init_log_alpha", np.log(max(self.base_lr, 1e-6))
                ),
                meta_type=self.meta_config.get("type", "idbd"),
                min_alpha=self.meta_config.get("min_alpha", 1e-6),
                max_alpha=self.meta_config.get("max_alpha", 1.0),
            )
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.base_lr)
            self.meta_updater = None

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
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        elif isinstance(action, torch.Tensor):
            action = action.to(self.device)
        else:
            action = torch.as_tensor(action, dtype=torch.long, device=self.device)
        if isinstance(reward, np.ndarray):
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        elif isinstance(reward, torch.Tensor):
            reward = reward.to(self.device)
        else:
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        elif isinstance(next_state, torch.Tensor):
            next_state = next_state.to(self.device)
        else:
            next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        if isinstance(done, np.ndarray):
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        elif isinstance(done, torch.Tensor):
            done = done.to(self.device)
        else:
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

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
            if self._state_encoder is not None:
                with torch.no_grad():
                    feature_state = self._state_encoder.encode_tensor(state)
            else:
                feature_state = state
            step_sizes = self.meta_updater.update_step_sizes(gradients, feature_state)
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

    def to(self, device):
        self.device = _resolve_device(device)
        self.q_net.to(self.device)
        self.target_net.to(self.device)
        return self

    def get_average_td_error(self, n=100):
        """Get average TD error over last n updates"""
        if len(self.td_errors) == 0:
            return 0.0
        recent_errors = self.td_errors[-n:]
        return np.mean(recent_errors)
