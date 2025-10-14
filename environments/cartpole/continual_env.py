"""
Continual Learning CartPole Environment
Supports dynamic regime switching for non-stationary RL research
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Optional, Tuple, List
from environments.base_env import OaKEnvironment


class ContinualCartPoleEnv(OaKEnvironment):
    """
    CartPole environment with regime switching for continual learning experiments.

    Regimes modify physics parameters to create distribution shift:
    - R1 (Base): Standard CartPole-v1 parameters
    - R2 (Heavy): Increased pole mass (0.1 → 0.5)
    - R3 (Long): Increased pole length (0.5 → 1.0)
    - R4 (Friction): Added cart and pole friction
    - R5 (Gravity): Increased gravity (9.8 → 12.0)

    State: (x, x_dot, theta, theta_dot)
    Actions: {0: Left, 1: Right}
    Reward: +1 per step until failure
    """

    # Regime definitions (parameters that differ from base)
    REGIMES = {
        'R1_base': {
            'gravity': 9.8,
            'masscart': 1.0,
            'masspole': 0.1,
            'length': 0.5,
            'force_mag': 10.0,
        },
        'R2_heavy': {
            'masspole': 0.5,  # 5x heavier pole
        },
        'R3_long': {
            'length': 1.0,  # 2x longer pole
        },
        'R4_friction': {
            # Note: Gymnasium CartPole doesn't have friction params built-in
            # We'll simulate by adding damping to velocities
            'velocity_damping': 0.95,  # multiply velocities by this each step
        },
        'R5_gravity': {
            'gravity': 12.0,  # ~1.22x Earth gravity
        },
    }

    # Base CartPole-v1 parameters (for reference)
    BASE_PARAMS = {
        'gravity': 9.8,
        'masscart': 1.0,
        'masspole': 0.1,
        'length': 0.5,  # half-pole length in gym is 0.5
        'force_mag': 10.0,
    }

    def __init__(self, regime: str = 'R1_base'):
        """
        Initialize environment with specified regime.

        Args:
            regime: One of ['R1_base', 'R2_heavy', 'R3_long', 'R4_friction', 'R5_gravity']
        """
        super().__init__()
        self.env = gym.make("CartPole-v1")
        self._state_dim = 4  # (x, x_dot, theta, theta_dot)
        self._action_dim = 2  # Left, Right

        # Track current regime
        self.current_regime = None
        self.regime_history: List[Tuple[int, str]] = []  # (step, regime_name)
        self.regime_transition_steps: List[int] = []

        # Apply initial regime
        self.switch_regime(regime)

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def action_dim(self):
        return self._action_dim

    def switch_regime(self, regime: str) -> None:
        """
        Switch to a new regime by modifying environment physics.

        Args:
            regime: Target regime name
        """
        if regime not in self.REGIMES:
            raise ValueError(f"Unknown regime: {regime}. Must be one of {list(self.REGIMES.keys())}")

        # Log regime transition
        if self.current_regime is not None and self.current_regime != regime:
            self.regime_transition_steps.append(self.step_count)

        self.current_regime = regime
        self.regime_history.append((self.step_count, regime))

        # Start with base parameters
        params = self.BASE_PARAMS.copy()

        # Apply regime-specific modifications
        regime_params = self.REGIMES[regime]
        params.update(regime_params)

        # Update gymnasium CartPole physics
        # Access the underlying CartPole environment
        unwrapped = self.env.unwrapped

        # Set physics parameters
        if 'gravity' in params:
            unwrapped.gravity = params['gravity']
        if 'masscart' in params:
            unwrapped.masscart = params['masscart']
        if 'masspole' in params:
            unwrapped.masspole = params['masspole']
        if 'length' in params:
            unwrapped.length = params['length']
        if 'force_mag' in params:
            unwrapped.force_mag = params['force_mag']

        # Recompute derived quantities
        unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
        unwrapped.polemass_length = unwrapped.masspole * unwrapped.length

        # Store velocity damping for friction regime
        self.velocity_damping = params.get('velocity_damping', 1.0)

        print(f"[Regime Switch] Step {self.step_count}: {self.current_regime}")
        print(f"  Physics: gravity={unwrapped.gravity:.1f}, "
              f"masspole={unwrapped.masspole:.2f}, length={unwrapped.length:.2f}")

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        self.current_state, _ = self.env.reset()
        self.step_count = 0
        return self.current_state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment.

        Returns: (next_state, reward, done, info)
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        # Apply friction (velocity damping) if in friction regime
        if self.velocity_damping < 1.0:
            next_state[1] *= self.velocity_damping  # x_dot
            next_state[3] *= self.velocity_damping  # theta_dot

        self.current_state = next_state
        self.step_count += 1

        # Add regime info
        info['regime'] = self.current_regime
        info['regime_step'] = self.step_count - (
            self.regime_transition_steps[-1] if self.regime_transition_steps else 0
        )

        return next_state.copy(), reward, done, info

    def get_state_components(self, state: Optional[np.ndarray] = None) -> Tuple[float, float, float, float]:
        """
        Extract individual state components for GVF cumulants.

        Returns: (x, x_dot, theta, theta_dot)
        """
        if state is None:
            state = self.current_state
        x, x_dot, theta, theta_dot = state
        return x, x_dot, theta, theta_dot

    def get_regime_info(self) -> Dict:
        """Get current regime information"""
        return {
            'current_regime': self.current_regime,
            'regime_step': self.step_count,
            'num_transitions': len(self.regime_transition_steps),
            'transition_steps': self.regime_transition_steps.copy(),
            'regime_history': self.regime_history.copy(),
        }

    def close(self):
        """Close environment"""
        self.env.close()


def create_regime_schedule(
    episodes_per_regime: int = 100,
    regimes: Optional[List[str]] = None
) -> List[Tuple[str, int, int]]:
    """
    Create a regime schedule for continual learning.

    Args:
        episodes_per_regime: Number of episodes per regime
        regimes: List of regime names (default: all 5 regimes)

    Returns:
        List of (regime_name, start_episode, end_episode) tuples
    """
    if regimes is None:
        regimes = ['R1_base', 'R2_heavy', 'R3_long', 'R4_friction', 'R5_gravity']

    schedule = []
    for i, regime in enumerate(regimes):
        start_ep = i * episodes_per_regime
        end_ep = (i + 1) * episodes_per_regime
        schedule.append((regime, start_ep, end_ep))

    return schedule
