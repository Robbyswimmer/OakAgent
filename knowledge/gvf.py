"""
Generalized Value Functions (GVFs) - Knowledge Layer
Predictive knowledge as learned GVFs (OaK core principle #2)

This module provides the generic GVF base class.
Environment-specific GVF implementations should inherit from this class
and implement the compute_cumulant() method.

Example (CartPole):
    See environments/cartpole/gvfs.py for UprignessGVF, CenteringGVF, etc.

Example (ARC):
    See environments/arc/gvfs.py for GridSimilarityGVF, EntropyGVF, etc.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from meta.meta_optimizer import MetaOptimizerAdapter

class GVF(nn.Module):
    """
    Base Generalized Value Function
    Learns to predict E[cumulant] using TD learning

    Subclasses must implement:
        compute_cumulant(state) -> float: Returns the cumulant value for the given state
    """

    def __init__(self, state_dim, hidden_size=64, gamma=0.97, use_idbd=True):
        super().__init__()
        self.state_dim = state_dim
        self.gamma = gamma
        self.use_idbd = use_idbd
        self.base_lr = 1e-3

        # Function approximator (small MLP)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize with SMALL weights for normalized cumulants [0,1]
        # Default init gives predictions ~500-1000, but we want ~0-5
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Eligibility traces for TD(lambda)
        self.traces = {name: torch.zeros_like(param)
                      for name, param in self.net.named_parameters()}

        # Prediction history for feature mining
        self.prediction_history = deque(maxlen=1000)

        # Prediction error tracking
        self.error_history = deque(maxlen=100)
        self.normalized_error_history = deque(maxlen=100)

        # Running statistics of TD targets (for normalization)
        self._target_mean = 0.0
        self._target_m2 = 0.0
        self._target_count = 0

        self.optimizer = None

    def configure_optimizer(self, base_lr=1e-3, meta_config=None):
        """Attach optimizer/meta-optimizer for GD updates."""
        self.base_lr = base_lr
        self.optimizer = MetaOptimizerAdapter(self.net.parameters(), base_lr=base_lr, meta_config=meta_config)

    def predict(self, state):
        """Predict value for given state"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            prediction = self.net(state).squeeze()

        result = prediction.item() if prediction.dim() == 0 else prediction.cpu().numpy()

        # Return 0.0 if prediction is invalid (shouldn't happen with safeguards, but just in case)
        if not np.isfinite(result).all() if isinstance(result, np.ndarray) else not np.isfinite(result):
            return 0.0

        # Normalize by theoretical max: cumulant_max / (1 - gamma)
        # For normalized cumulants in [0,1], max prediction = 1/(1-gamma)
        normalization_factor = 1.0 / max(1.0 - self.gamma, 0.01)  # avoid div by zero
        result = result / normalization_factor

        return result

    def forward(self, state):
        """Forward pass (for training)"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        prediction = self.net(state).squeeze(-1)
        return torch.clamp(prediction, -5.0, 5.0)

    def compute_td_error(self, state, cumulant, next_state, gamma=None):
        """
        Compute TD error: delta = c + gamma * V(s') - V(s)
        where c is the cumulant
        """
        if gamma is None:
            gamma = self.gamma

        v_s = self.forward(state)
        with torch.no_grad():
            v_s_next = self.forward(next_state)

        td_error = cumulant + gamma * v_s_next - v_s

        return td_error, v_s

    def update(self, state, cumulant, next_state, gamma=None, step_size=None):
        """
        TD update: V(s) <- V(s) + alpha * [c + gamma * V(s') - V(s)]

        Returns: td_error (for meta-learning)
        """
        if gamma is None:
            gamma = self.gamma

        # Compute TD error
        td_error, v_s = self.compute_td_error(state, cumulant, next_state, gamma)

        # Compute loss
        loss = td_error.pow(2).mean()

        # Backward pass
        self.net.zero_grad(set_to_none=True)
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients for stability (relaxed to track non-stationary targets)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.5)

        # Apply gradient (with optional custom step-size override)
        if step_size is not None:
            step_size = np.clip(step_size, 1e-6, 0.1)
            with torch.no_grad():
                for param in self.net.parameters():
                    if param.grad is not None:
                        param.data -= step_size * param.grad
                        param.data.clamp_(-10.0, 10.0)
        elif self.optimizer is not None:
            feature_vec = self._feature_vector_from_state(state)
            self.optimizer.step(feature_vec, clip_range=(-10.0, 10.0))
        else:
            with torch.no_grad():
                for param in self.net.parameters():
                    if param.grad is not None:
                        param.data -= self.base_lr * param.grad
                        param.data.clamp_(-10.0, 10.0)

        # Check for NaN and reset if necessary
        has_nan = False
        with torch.no_grad():
            for param in self.net.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    has_nan = True
                    break

        if has_nan:
            # Reset network weights
            self._reset_weights()
            return 0.0

        # Track error
        td_error_val = float(td_error.detach().mean().item())
        if np.isfinite(td_error_val):
            self.error_history.append(abs(td_error_val))

            # Update running target stats and normalized error
            with torch.no_grad():
                target = (cumulant + (gamma if gamma is not None else self.gamma) * self.forward(next_state)).detach()
            target_val = float(target.mean().item())
            self._update_target_stats(target_val)
            target_std = self._get_target_std()
            if target_std > 0:
                self.normalized_error_history.append(abs(td_error_val) / target_std)
            else:
                self.normalized_error_history.append(abs(td_error_val))

        # Store prediction
        v_s_val = float(v_s.detach().mean().item())
        if np.isfinite(v_s_val):
            self.prediction_history.append(v_s_val)

        return td_error_val

    def _update_target_stats(self, value):
        """Online Welford update for target statistics."""
        self._target_count += 1
        delta = value - self._target_mean
        self._target_mean += delta / self._target_count
        delta2 = value - self._target_mean
        self._target_m2 += delta * delta2

    def _get_target_std(self):
        if self._target_count < 2:
            return 1.0
        variance = self._target_m2 / (self._target_count - 1)
        return math.sqrt(max(variance, 1e-6))

    def _reset_weights(self):
        """Reset network weights to small random values"""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _feature_vector_from_state(state):
        if isinstance(state, torch.Tensor):
            vector = state.detach().cpu().numpy()
        else:
            vector = np.asarray(state, dtype=np.float32)
        if vector.ndim > 1:
            vector = vector.reshape(-1)
        return vector

    def get_recent_predictions(self, n=100):
        """Get last n predictions for feature mining"""
        history = list(self.prediction_history)
        return history[-n:] if len(history) >= n else history

    def get_average_error(self):
        """Get average prediction error over recent history"""
        if len(self.error_history) == 0:
            return 0.0
        return np.mean(list(self.error_history))

    def get_normalized_error(self):
        """Get normalized TD error (divided by target std)."""
        if len(self.normalized_error_history) == 0:
            return 0.0
        return float(np.mean(list(self.normalized_error_history)))

    def compute_cumulant(self, state):
        """
        Compute the cumulant for this GVF.
        Must be implemented by subclasses.

        Args:
            state: Environment state (numpy array or torch tensor)

        Returns:
            cumulant: Scalar value representing the cumulant

        Example (CartPole uprightness):
            return abs(state[2]) / 0.2095  # |theta| normalized

        Example (ARC grid similarity):
            return np.mean(current_grid == solution_grid)
        """
        raise NotImplementedError("Subclasses must implement compute_cumulant()")
