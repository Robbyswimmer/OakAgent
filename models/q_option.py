"""
Option Q-Function (SMDP Q-Learning)
Value function for options (temporal abstractions)
Uses SMDP discounting: gamma^tau where tau is option duration
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

class OptionQNetwork(nn.Module):
    """Q-network for options with optional state encoder"""

    def __init__(self, state_dim, num_options, hidden_size=128, state_encoder=None, device=None):
        super().__init__()
        self.state_dim = state_dim
        self.num_options = num_options
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
            nn.Linear(hidden_size, num_options)
        )

        self.to(self.device)

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) or (state_dim,)

        Returns:
            q_values: (batch, num_options) or (num_options,)
        """
        if isinstance(state, np.ndarray):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        elif isinstance(state, torch.Tensor):
            state = state.to(self.device)
        else:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        encoder = getattr(self, "_state_encoder", None)
        if encoder is not None:
            with torch.no_grad():
                state = encoder.encode_tensor(state)

        output = self.net(state)
        if output.dim() == 2 and output.size(0) == 1:
            return output.squeeze(0)
        return output


class SMDPQNetwork:
    """
    SMDP Q-Learning for options
    Q_O(s, o) <- Q_O(s, o) + alpha * [R_o + gamma^tau * max_o' Q_O(s', o') - Q_O(s, o)]
    """

    def __init__(
        self,
        state_dim,
        num_options,
        gamma=0.99,
        lr=1e-3,
        meta_config=None,
        state_encoder=None,
        latent_dim=None,
        device=None,
    ):
        self.state_dim = state_dim
        self.num_options = num_options
        self.gamma = gamma
        self.base_lr = lr
        self.device = _resolve_device(device)
        if isinstance(state_encoder, nn.Module):
            object.__setattr__(self, '_state_encoder', state_encoder)
        else:
            self._state_encoder = state_encoder

        if latent_dim is None:
            latent_dim = state_dim
        self.latent_dim = latent_dim

        # Q-network (may be None until first option exists)
        self.q_net = None

        self.meta_config = meta_config or {}
        self.use_meta = meta_config is not None and not self.meta_config.get(
            "disabled", False
        )
        self.meta_updater = None
        self.optimizer = None
        if num_options > 0:
            self._initialize_network(num_options)

        # Tracking
        self.td_errors = []
        self.option_counts = {i: 0 for i in range(num_options)}

    def _initialize_network(self, num_options):
        """Create Q-network and optimizer/meta-learner for current option count."""
        self.q_net = OptionQNetwork(self.latent_dim, num_options, state_encoder=self._state_encoder, device=self.device)
        if self.use_meta:
            self._reinit_meta()
            self.optimizer = None
        else:
            self.meta_updater = None
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.base_lr)

    def predict(self, state):
        """Get Q-values for all options at state"""
        if self.q_net is None or self.num_options == 0:
            return np.zeros(0, dtype=np.float32)

        with torch.no_grad():
            return self.q_net(state).cpu().numpy()

    def select_option(self, state, epsilon=0.0):
        """Epsilon-greedy option selection"""
        if self.num_options == 0 or self.q_net is None:
            return None

        if np.random.rand() < epsilon:
            return np.random.randint(self.num_options)

        q_values = self.predict(state)
        if q_values.size == 0:
            return None
        return int(np.argmax(q_values))

    def update_smdp(self, s_start, option_id, R_total, s_end, duration, done=False):
        """
        SMDP TD update for option execution

        Args:
            s_start: starting state of option
            option_id: which option was executed
            R_total: cumulative reward during option
            s_end: ending state of option
            duration: number of steps (tau)
            done: whether episode ended

        Returns:
            loss value
        """
        if self.q_net is None or self.num_options == 0:
            return 0.0

        # Convert to tensors
        if isinstance(s_start, np.ndarray):
            s_start = torch.as_tensor(s_start, dtype=torch.float32, device=self.device)
        elif isinstance(s_start, torch.Tensor):
            s_start = s_start.to(self.device)
        else:
            s_start = torch.as_tensor(s_start, dtype=torch.float32, device=self.device)
        if isinstance(s_end, np.ndarray):
            s_end = torch.as_tensor(s_end, dtype=torch.float32, device=self.device)
        elif isinstance(s_end, torch.Tensor):
            s_end = s_end.to(self.device)
        else:
            s_end = torch.as_tensor(s_end, dtype=torch.float32, device=self.device)

        if s_start.dim() == 1:
            s_start = s_start.unsqueeze(0)
            s_end = s_end.unsqueeze(0)

        option_id = torch.tensor([option_id], dtype=torch.long, device=self.device)
        R_total = torch.tensor([R_total], dtype=torch.float32, device=self.device)
        duration = torch.tensor([duration], dtype=torch.float32, device=self.device)

        # Current Q-value: Q(s, o)
        q_values = self.q_net(s_start)
        if q_values.dim() == 1:
            q_values = q_values.unsqueeze(0)
        q_value = q_values.gather(1, option_id.unsqueeze(1)).squeeze(1)

        # Target Q-value with SMDP discounting
        with torch.no_grad():
            next_q_values = self.q_net(s_end)
            if next_q_values.dim() == 1:
                next_q_values = next_q_values.unsqueeze(0)
            next_q_value = next_q_values.max(dim=1)[0]

            # SMDP target: R + gamma^tau * max_o' Q(s', o')
            gamma_tau = self.gamma ** duration.float()
            target_q_value = R_total + gamma_tau * next_q_value * (1 - float(done))

        # Loss
        loss = F.mse_loss(q_value, target_q_value)

        if self.use_meta and self.meta_updater is not None:
            self.q_net.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [param.grad for param in self.q_net.parameters()]
            if self._state_encoder is not None:
                with torch.no_grad():
                    feature_state = self._state_encoder.encode_tensor(s_start)
                feature_state = feature_state.reshape(-1)
                if feature_state.numel() == 0:
                    feature_state = torch.ones(1, dtype=torch.float32, device=self.device)
            else:
                feature_state = torch.ones(1, dtype=torch.float32, device=self.device)
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

        # Track option usage
        self.option_counts[option_id.item()] += 1

        return loss.item()

    def to(self, device):
        self.device = _resolve_device(device)
        if self.q_net is not None:
            self.q_net.to(self.device)
        return self

    def update_from_trajectory(self, trajectory, option_id):
        """
        Update from full option trajectory

        Args:
            trajectory: list of (s, a, r, s', done) tuples
            option_id: which option was executed
        """
        if len(trajectory) == 0:
            return 0.0

        s_start = trajectory[0][0]
        s_end = trajectory[-1][3]
        R_total = sum([t[2] for t in trajectory])
        duration = len(trajectory)
        done = trajectory[-1][4]

        return self.update_smdp(s_start, option_id, R_total, s_end, duration, done)

    def get_option_usage(self):
        """Get option usage statistics"""
        return self.option_counts.copy()

    def get_average_td_error(self, n=100):
        """Get average TD error over last n updates"""
        if len(self.td_errors) == 0:
            return 0.0
        recent_errors = self.td_errors[-n:]
        return np.mean(recent_errors)

    def _reinit_meta(self):
        if not self.use_meta or self.q_net is None:
            return

        self.meta_updater = TorchIDBD(
            self.q_net.parameters(),
            mu=self.meta_config.get("mu", 1e-3),
            init_log_alpha=self.meta_config.get("init_log_alpha", np.log(self.base_lr)),
            meta_type=self.meta_config.get("type", "idbd"),
            min_alpha=self.meta_config.get("min_alpha", 1e-6),
            max_alpha=self.meta_config.get("max_alpha", 1.0),
        )

    def add_option(self):
        """Dynamically add a new option (for FC-STOMP)"""
        old_num = self.num_options
        self.num_options += 1

        if old_num == 0 or self.q_net is None:
            # First option – just initialize new network
            self._initialize_network(self.num_options)
        else:
            old_state = self.q_net.state_dict()
            new_net = OptionQNetwork(self.latent_dim, self.num_options, state_encoder=self._state_encoder, device=self.device)
            new_state = new_net.state_dict()

            # Copy shared layers
            for name, param in old_state.items():
                if name not in new_state:
                    continue
                if (
                    name.endswith("weight")
                    and new_state[name].shape[0] == self.num_options
                    and param.shape[0] == old_num
                ):
                    new_state[name][:old_num] = param
                elif (
                    name.endswith("bias")
                    and new_state[name].shape[0] == self.num_options
                    and param.shape[0] == old_num
                ):
                    new_state[name][:old_num] = param
                elif new_state[name].shape == param.shape:
                    new_state[name] = param

            new_net.load_state_dict(new_state)
            self.q_net = new_net

            if self.use_meta:
                self._reinit_meta()
                self.optimizer = None
            else:
                self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.base_lr)

        self.option_counts[self.num_options - 1] = 0

    def reset_option(self, option_id):
        """Reinitialize an existing option head without changing network size."""
        if option_id < 0 or option_id >= self.num_options:
            return

        if self.q_net is None:
            return

        last_layer = None
        for layer in self.q_net.net:
            if isinstance(layer, nn.Linear):
                last_layer = layer

        if last_layer is not None and option_id < last_layer.weight.shape[0]:
            with torch.no_grad():
                nn.init.zeros_(last_layer.weight[option_id])
                if last_layer.bias is not None:
                    last_layer.bias[option_id] = 0.0

        self.option_counts[option_id] = 0

        # Refresh meta learner to avoid stale step-size state
        if self.use_meta:
            self._reinit_meta()
