"""
Option-Level SMDP Models
Learns p(s', R, τ | s, o) for each option
Enables multi-timescale planning (OaK FC-STOMP stage O)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.meta_optimizer import MetaOptimizerAdapter

class OptionModel(nn.Module):
    """
    SMDP model for a single option
    Predicts: (Δs, R_total, duration) from start state
    """

    def __init__(self, state_dim, hidden_size=128):
        super().__init__()
        self.state_dim = state_dim

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Output heads
        self.delta_head = nn.Linear(hidden_size, state_dim)  # Δs prediction
        self.reward_head = nn.Linear(hidden_size, 1)  # Total reward R_o
        self.duration_head = nn.Linear(hidden_size, 1)  # Duration τ

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim) or (state_dim,)

        Returns:
            delta_state, total_reward, duration
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        features = self.shared(state)

        delta_state = self.delta_head(features)
        total_reward = self.reward_head(features)
        duration = F.softplus(self.duration_head(features)) + 1.0  # ensure τ >= 1

        return delta_state, total_reward, duration

    def predict(self, state):
        """Predict SMDP outcome: (s', R, τ)"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)

            delta_s, R, tau = self.forward(state)
            next_state = state + delta_s

            return (
                next_state.squeeze().cpu().numpy(),
                R.squeeze().item(),
                int(tau.squeeze().item())
            )


class OptionModelLibrary:
    """
    Collection of option models
    Maps option_id -> OptionModel
    """

    def __init__(
        self,
        state_dim,
        hidden_size=128,
        lr=1e-3,
        min_rollouts=5,
        error_threshold=None,
        meta_config=None,
    ):
        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.min_rollouts = min_rollouts
        self.error_threshold = error_threshold
        self.meta_config = meta_config.copy() if meta_config is not None else None

        # Dictionary: option_id -> (model, optimizer)
        self.models = {}
        self.prediction_errors = {}
        self.experience_counts = {}

    def add_option(self, option_id):
        """Add a new option model"""
        model = OptionModel(self.state_dim, self.hidden_size)
        optimizer = MetaOptimizerAdapter(model.parameters(), base_lr=self.lr, meta_config=self.meta_config)
        self.models[option_id] = (model, optimizer)
        self.prediction_errors[option_id] = []
        self.experience_counts[option_id] = 0

    def remove_option(self, option_id):
        """Remove model artifacts for an option."""
        self.models.pop(option_id, None)
        self.prediction_errors.pop(option_id, None)
        self.experience_counts.pop(option_id, None)

    def predict(self, option_id, state):
        """Predict SMDP outcome for option"""
        if option_id not in self.models:
            # If option doesn't exist, return default prediction
            return state, 0.0, 1

        model, _ = self.models[option_id]
        if not self.is_trained(option_id):
            if isinstance(state, np.ndarray):
                next_state = state.copy()
            else:
                next_state = np.array(state, dtype=np.float32)
            return next_state, 0.0, 1
        return model.predict(state)

    def fit_from_rollout(self, option_id, s_start, s_end, R_total, duration):
        """
        Update option model from a single rollout

        Args:
            option_id: option identifier
            s_start: starting state
            s_end: ending state
            R_total: cumulative reward
            duration: number of steps (τ)
        """
        if option_id not in self.models:
            self.add_option(option_id)

        model, optimizer = self.models[option_id]

        # Convert to tensors
        if isinstance(s_start, np.ndarray):
            s_start = torch.FloatTensor(s_start)
        if isinstance(s_end, np.ndarray):
            s_end = torch.FloatTensor(s_end)

        if s_start.dim() == 1:
            s_start = s_start.unsqueeze(0)
            s_end = s_end.unsqueeze(0)

        # Target values
        target_delta = s_end - s_start
        target_reward = torch.FloatTensor([[R_total]])
        target_duration = torch.FloatTensor([[duration]])

        # Forward pass
        pred_delta, pred_reward, pred_duration = model(s_start)

        # Loss
        loss_delta = F.mse_loss(pred_delta, target_delta)
        loss_reward = F.mse_loss(pred_reward, target_reward)
        loss_duration = F.mse_loss(pred_duration, target_duration)
        loss = loss_delta + loss_reward + 0.1 * loss_duration  # weight duration less

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        feature_vec = s_start.detach().cpu().numpy().reshape(-1)
        optimizer.step(feature_vec, clip_range=(-10.0, 10.0))

        # Track error
        self.prediction_errors[option_id].append(loss.item())
        self.experience_counts[option_id] += 1

        return loss.item()

    def fit_from_trajectory(self, option_id, trajectory):
        """
        Update option model from full trajectory

        Args:
            trajectory: list of (s, a, r, s', done) tuples
        """
        if len(trajectory) == 0:
            return 0.0

        # Extract SMDP quantities
        s_start = trajectory[0][0]
        s_end = trajectory[-1][3]  # s' of last step
        R_total = sum([t[2] for t in trajectory])
        duration = len(trajectory)

        return self.fit_from_rollout(option_id, s_start, s_end, R_total, duration)

    def get_average_error(self, option_id, n=100):
        """Get average prediction error for option"""
        if option_id not in self.prediction_errors:
            return 0.0

        errors = self.prediction_errors[option_id]
        if len(errors) == 0:
            return 0.0

        recent_errors = errors[-n:]
        return np.mean(recent_errors)

    def get_all_option_ids(self):
        """Get list of all option IDs"""
        return sorted(self.models.keys())

    def is_trained(self, option_id):
        """Check whether an option model has sufficient data to be trustworthy."""
        count = self.experience_counts.get(option_id, 0)
        if count < self.min_rollouts:
            return False

        if self.error_threshold is None:
            return True

        avg_error = self.get_average_error(option_id, n=self.min_rollouts)
        return avg_error <= self.error_threshold
