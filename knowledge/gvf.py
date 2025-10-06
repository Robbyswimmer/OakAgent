"""
Generalized Value Functions (GVFs) - Knowledge Layer
Predictive knowledge as learned GVFs (OaK core principle #2)

Each GVF predicts a specific cumulant under a policy:
- g1: E[|theta|] - uprightness predictor
- g2: E[|x|] - centering predictor
- g3: E[|theta_dot| + |x_dot|] - stability predictor
- g4: E[time-to-failure] - survival horizon predictor
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

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

        # Function approximator (small MLP)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Eligibility traces for TD(lambda)
        self.traces = {name: torch.zeros_like(param)
                      for name, param in self.net.named_parameters()}

        # Prediction history for feature mining
        self.prediction_history = deque(maxlen=1000)

        # Prediction error tracking
        self.error_history = deque(maxlen=100)

    def predict(self, state):
        """Predict value for given state"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            prediction = self.net(state).squeeze()

        return prediction.item() if prediction.dim() == 0 else prediction.cpu().numpy()

    def forward(self, state):
        """Forward pass (for training)"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        return self.net(state).squeeze(-1)

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
        self.net.zero_grad()
        loss.backward()

        # Apply gradient (with optional custom step-size from IDBD)
        if step_size is not None:
            with torch.no_grad():
                for param in self.net.parameters():
                    if param.grad is not None:
                        param.data -= step_size * param.grad
        else:
            # Use default optimizer
            with torch.no_grad():
                for param in self.net.parameters():
                    if param.grad is not None:
                        param.data -= 0.001 * param.grad

        # Track error
        self.error_history.append(abs(td_error.item()))

        # Store prediction
        self.prediction_history.append(v_s.item())

        return td_error.item()

    def get_recent_predictions(self, n=100):
        """Get last n predictions for feature mining"""
        history = list(self.prediction_history)
        return history[-n:] if len(history) >= n else history

    def get_average_error(self):
        """Get average prediction error over recent history"""
        if len(self.error_history) == 0:
            return 0.0
        return np.mean(list(self.error_history))


class UprignessGVF(GVF):
    """g1: Predicts E[|theta|] - how upright the pole is"""

    def __init__(self, state_dim, hidden_size=64, gamma=0.97):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "uprightness"

    def compute_cumulant(self, state):
        """Cumulant: |theta|"""
        if isinstance(state, np.ndarray):
            theta = state[2]  # theta is 3rd component
        else:
            theta = state[..., 2]
        return abs(theta)


class CenteringGVF(GVF):
    """g2: Predicts E[|x|] - how centered the cart is"""

    def __init__(self, state_dim, hidden_size=64, gamma=0.97):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "centering"

    def compute_cumulant(self, state):
        """Cumulant: |x|"""
        if isinstance(state, np.ndarray):
            x = state[0]  # x is 1st component
        else:
            x = state[..., 0]
        return abs(x)


class StabilityGVF(GVF):
    """g3: Predicts E[|theta_dot| + |x_dot|] - velocity stability"""

    def __init__(self, state_dim, hidden_size=64, gamma=0.97):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "stability"

    def compute_cumulant(self, state):
        """Cumulant: |theta_dot| + |x_dot|"""
        if isinstance(state, np.ndarray):
            x_dot = state[1]
            theta_dot = state[3]
        else:
            x_dot = state[..., 1]
            theta_dot = state[..., 3]
        return abs(theta_dot) + abs(x_dot)


class SurvivalGVF(GVF):
    """g4: Predicts E[time-to-failure] - survival horizon"""

    def __init__(self, state_dim, hidden_size=64, gamma=0.999):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "survival"

    def compute_cumulant(self, state):
        """Cumulant: 1 (cumulative return until termination)"""
        return 1.0


class HordeGVFs:
    """
    Horde of GVFs - collection of all predictive knowledge
    Updates all GVFs in parallel (OaK continual learning)
    """

    def __init__(self, state_dim, config):
        self.state_dim = state_dim

        # Create the 4 core GVFs
        self.gvfs = {
            'uprightness': UprignessGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_SHORT),
            'centering': CenteringGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_SHORT),
            'stability': StabilityGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_SHORT),
            'survival': SurvivalGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_LONG)
        }

        self.config = config

    def predict_all(self, state):
        """Get predictions from all GVFs"""
        predictions = {}
        for name, gvf in self.gvfs.items():
            predictions[name] = gvf.predict(state)
        return predictions

    def update_all(self, state, next_state, step_sizes=None):
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
            td_error = gvf.update(state, cumulant, next_state, step_size=step_size)
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

    def __getitem__(self, name):
        """Access GVF by name"""
        return self.gvfs[name]
