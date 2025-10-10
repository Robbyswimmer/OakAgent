"""
Generalized Value Functions (GVFs) - Knowledge Layer
Predictive knowledge as learned GVFs (OaK core principle #2)

Each GVF predicts a specific cumulant under a policy:
- g1: E[|theta|] - uprightness predictor
- g2: E[|x|] - centering predictor
- g3: E[|theta_dot| + |x_dot|] - stability predictor
- g4: E[time-to-failure] - survival horizon predictor
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

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

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


class UprignessGVF(GVF):
    """g1: Predicts E[|theta|] - how upright the pole is"""

    def __init__(self, state_dim, hidden_size=64, gamma=0.97):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "uprightness"
        self.theta_max = 0.2095  # ~12 degrees (CartPole termination threshold)

    def compute_cumulant(self, state):
        """Cumulant: normalized |theta| ∈ [0, 1]"""
        if isinstance(state, np.ndarray):
            theta = state[2]  # theta is 3rd component
        else:
            theta = state[..., 2]
        return abs(theta) / self.theta_max


class CenteringGVF(GVF):
    """g2: Predicts E[|x|] - how centered the cart is"""

    def __init__(self, state_dim, hidden_size=64, gamma=0.97):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "centering"
        self.x_max = 2.4  # CartPole termination threshold

    def compute_cumulant(self, state):
        """Cumulant: normalized |x| ∈ [0, 1]"""
        if isinstance(state, np.ndarray):
            x = state[0]  # x is 1st component
        else:
            x = state[..., 0]
        return abs(x) / self.x_max


class StabilityGVF(GVF):
    """g3: Predicts E[|theta_dot| + |x_dot|] - velocity stability"""

    def __init__(self, state_dim, hidden_size=64, gamma=0.97):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "stability"
        self.velocity_scale = 5.0  # Empirical max for |theta_dot| + |x_dot|

    def compute_cumulant(self, state):
        """Cumulant: normalized velocity magnitude ∈ [0, 1]"""
        if isinstance(state, np.ndarray):
            x_dot = state[1]
            theta_dot = state[3]
        else:
            x_dot = state[..., 1]
            theta_dot = state[..., 3]
        total_velocity = abs(theta_dot) + abs(x_dot)
        return min(total_velocity / self.velocity_scale, 1.0)


class SurvivalGVF(GVF):
    """g4: Predicts E[time-to-failure] - survival horizon"""

    def __init__(self, state_dim, hidden_size=64, gamma=0.99, horizon=200.0):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "survival"
        self.horizon = horizon

    def compute_cumulant(self, state):
        """Cumulant: normalized survival reward (≈1/horizon)."""
        return 1.0 / self.horizon


class HordeGVFs:
    """
    Horde of GVFs - collection of all predictive knowledge
    Updates all GVFs in parallel (OaK continual learning)
    """

    def __init__(self, state_dim, config, meta_config=None):
        self.state_dim = state_dim

        # Create the 4 core GVFs
        self.gvfs = {
            'uprightness': UprignessGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_SHORT),
            'centering': CenteringGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_SHORT),
            'stability': StabilityGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_SHORT),
            'survival': SurvivalGVF(
                state_dim,
                config.GVF_HIDDEN_SIZE,
                config.GVF_GAMMA_LONG,
                horizon=getattr(config, 'FC_SURVIVAL_TARGET', 200.0),
            ),
        }

        self.config = config

        base_lr = getattr(config, 'GVF_LR', 1e-3)
        meta_cfg = None
        if meta_config is not None:
            meta_cfg = meta_config.copy()
            meta_cfg['init_log_alpha'] = math.log(base_lr)
        for gvf in self.gvfs.values():
            gvf.configure_optimizer(base_lr=base_lr, meta_config=meta_cfg)

    def predict_all(self, state):
        """Get predictions from all GVFs"""
        predictions = {}
        for name, gvf in self.gvfs.items():
            predictions[name] = gvf.predict(state)
        return predictions

    def update_all(self, state, next_state, done, step_sizes=None):
        """
        Update all GVFs with their respective cumulants
        (continual learning - called every step)

        Returns: dict of td_errors for meta-learning
        """
        td_errors = {}

        for name, gvf in self.gvfs.items():
            # Compute cumulant for this GVF
            cumulant = gvf.compute_cumulant(state)

            # Get step-size from IDBD if provided
            step_size = step_sizes.get(name) if step_sizes else None

            # Update GVF
            gamma = 0.0 if done else gvf.gamma
            td_error = gvf.update(state, cumulant, next_state, gamma=gamma, step_size=step_size)
            td_errors[name] = td_error

        return td_errors

    def get_feature_vector(self, state):
        """
        Get feature vector from GVF predictions
        This is the "knowledge" that drives option formation
        """
        predictions = self.predict_all(state)
        return np.array([predictions['uprightness'],
                        predictions['centering'],
                        predictions['stability'],
                        predictions['survival']])

    def get_prediction_histories(self):
        """Get recent predictions from all GVFs for feature mining"""
        histories = {}
        for name, gvf in self.gvfs.items():
            histories[name] = gvf.get_recent_predictions()
        return histories

    def get_average_errors(self):
        """Get average prediction errors for all GVFs"""
        errors = {}
        for name, gvf in self.gvfs.items():
            errors[name] = gvf.get_average_error()
        return errors

    def get_normalized_errors(self):
        """Get normalized TD errors for all GVFs."""
        normalized = {}
        for name, gvf in self.gvfs.items():
            normalized[name] = gvf.get_normalized_error()
        return normalized

    def __getitem__(self, name):
        """Access GVF by name"""
        return self.gvfs[name]
