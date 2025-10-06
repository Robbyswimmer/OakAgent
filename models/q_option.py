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

class OptionQNetwork(nn.Module):
    """Q-network for options"""

    def __init__(self, state_dim, num_options, hidden_size=128):
        super().__init__()
        self.state_dim = state_dim
        self.num_options = num_options

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_options)
        )

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) or (state_dim,)

        Returns:
            q_values: (batch, num_options) or (num_options,)
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            return self.net(state).squeeze(0)

        return self.net(state)


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
    ):
        self.state_dim = state_dim
        self.num_options = num_options
        self.gamma = gamma
        self.base_lr = lr

        # Q-network
        self.q_net = OptionQNetwork(state_dim, num_options)

        self.meta_config = meta_config or {}
        self.use_meta = meta_config is not None and not self.meta_config.get(
            "disabled", False
        )
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
        self.td_errors = []
        self.option_counts = {i: 0 for i in range(num_options)}

    def predict(self, state):
        """Get Q-values for all options at state"""
        with torch.no_grad():
            return self.q_net(state).cpu().numpy()

    def select_option(self, state, epsilon=0.0):
        """Epsilon-greedy option selection"""
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_options)

        q_values = self.predict(state)
        return np.argmax(q_values)

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
        # Convert to tensors
        if isinstance(s_start, np.ndarray):
            s_start = torch.FloatTensor(s_start)
        if isinstance(s_end, np.ndarray):
            s_end = torch.FloatTensor(s_end)

        if s_start.dim() == 1:
            s_start = s_start.unsqueeze(0)
            s_end = s_end.unsqueeze(0)

        option_id = torch.LongTensor([option_id])
        R_total = torch.FloatTensor([R_total])
        duration = torch.LongTensor([duration])

        # Current Q-value: Q(s, o)
        q_values = self.q_net(s_start)
        q_value = q_values.gather(1, option_id.unsqueeze(1)).squeeze(1)

        # Target Q-value with SMDP discounting
        with torch.no_grad():
            next_q_values = self.q_net(s_end)
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
            step_sizes = self.meta_updater.update_step_sizes(gradients, s_start)
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
        if self.use_meta:
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

        old_state = self.q_net.state_dict()
        new_net = OptionQNetwork(self.state_dim, self.num_options)
        new_state = new_net.state_dict()

        # Copy shared layers
        for name, param in old_state.items():
            if name not in new_state:
                continue
            if name.endswith("weight") and new_state[name].shape[0] == self.num_options and param.shape[0] == old_num:
                new_state[name][:old_num] = param
            elif name.endswith("bias") and new_state[name].shape[0] == self.num_options and param.shape[0] == old_num:
                new_state[name][:old_num] = param
            elif new_state[name].shape == param.shape:
                new_state[name] = param

        new_net.load_state_dict(new_state)
        self.q_net = new_net

        if self.use_meta:
            self._reinit_meta()
        else:
            self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.base_lr)

        self.option_counts[self.num_options - 1] = 0
