"""
Option Q-Function (SMDP Q-Learning)
Value function for options (temporal abstractions)
Uses SMDP discounting: gamma^tau where tau is option duration
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def __init__(self, state_dim, num_options, gamma=0.99, lr=1e-3):
        self.state_dim = state_dim
        self.num_options = num_options
        self.gamma = gamma

        # Q-network
        self.q_net = OptionQNetwork(state_dim, num_options)

        # Optimizer
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

        # Backward pass
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

    def add_option(self):
        """Dynamically add a new option (for FC-STOMP)"""
        self.num_options += 1

        # Create new network with additional output
        old_weights = self.q_net.state_dict()
        self.q_net = OptionQNetwork(self.state_dim, self.num_options)

        # Initialize new option weights
        # Copy old weights where possible
        new_weights = self.q_net.state_dict()
        for name, param in old_weights.items():
            if name in new_weights and param.shape == new_weights[name].shape:
                new_weights[name] = param

        self.q_net.load_state_dict(new_weights)

        # Update optimizer
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.optimizer.param_groups[0]['lr'])

        # Initialize count for new option
        self.option_counts[self.num_options - 1] = 0
