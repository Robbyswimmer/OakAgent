"""
CartPole-Specific GVF Definitions
Predictive knowledge layer for CartPole domain
"""
import math
import numpy as np
from knowledge.gvf import GVF


class UprignessGVF(GVF):
    """g1: Predicts E[|theta|] - how upright the pole is"""

    def __init__(self, state_dim, hidden_size=64, gamma=0.97, state_encoder=None, device=None):
        super().__init__(state_dim, hidden_size, gamma, state_encoder=state_encoder, device=device)
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

    def __init__(self, state_dim, hidden_size=64, gamma=0.97, state_encoder=None, device=None):
        super().__init__(state_dim, hidden_size, gamma, state_encoder=state_encoder, device=device)
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

    def __init__(self, state_dim, hidden_size=64, gamma=0.97, state_encoder=None, device=None):
        super().__init__(state_dim, hidden_size, gamma, state_encoder=state_encoder, device=device)
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

    def __init__(self, state_dim, hidden_size=64, gamma=0.99, horizon=200.0, state_encoder=None, device=None):
        super().__init__(state_dim, hidden_size, gamma, state_encoder=state_encoder, device=device)
        self.name = "survival"
        self.horizon = horizon

    def compute_cumulant(self, state):
        """Cumulant: 1.0 per step survived (will be normalized by 1/(1-γ) in predict)."""
        return 1.0


class CartPoleHordeGVFs:
    """
    Horde of GVFs for CartPole - collection of all predictive knowledge
    Updates all GVFs in parallel (OaK continual learning)
    """

    def __init__(self, state_dim, config, meta_config=None, state_encoder=None, device=None):
        self.state_dim = state_dim

        # Create the 4 core GVFs for CartPole
        self.gvfs = {
            'uprightness': UprignessGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_SHORT, state_encoder=state_encoder, device=device),
            'centering': CenteringGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_SHORT, state_encoder=state_encoder, device=device),
            'stability': StabilityGVF(state_dim, config.GVF_HIDDEN_SIZE, config.GVF_GAMMA_SHORT, state_encoder=state_encoder, device=device),
            'survival': SurvivalGVF(
                state_dim,
                config.GVF_HIDDEN_SIZE,
                config.GVF_GAMMA_LONG,
                horizon=getattr(config, 'FC_SURVIVAL_TARGET', 200.0),
                state_encoder=state_encoder,
                device=device,
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
        return {name: gvf.predict(state) for name, gvf in self.gvfs.items()}

    def update_all(self, state, next_state):
        """Update all GVFs with a transition"""
        for name, gvf in self.gvfs.items():
            cumulant = gvf.compute_cumulant(state)
            gvf.update(state, cumulant, next_state)

    def get_gvf(self, name):
        """Get a specific GVF by name"""
        return self.gvfs.get(name)

    def get_all_gvfs(self):
        """Get all GVFs"""
        return self.gvfs

    def get_average_errors(self):
        """Get average prediction error for each GVF"""
        return {name: gvf.get_average_error() for name, gvf in self.gvfs.items()}

    def get_normalized_errors(self):
        """Get normalized TD errors for each GVF"""
        return {name: gvf.get_normalized_error() for name, gvf in self.gvfs.items()}
