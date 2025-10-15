"""
Abstract Base Environment for OaK Framework
Defines the interface that all environments must implement
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class OaKEnvironment(ABC):
    """
    Abstract base class for OaK environments.

    All environments must provide:
    - State observation (continuous vector or grid)
    - Action execution
    - Reward signal
    - State components for GVF cumulants
    - Metadata about action/state spaces
    """

    def __init__(self):
        self.current_state: Optional[np.ndarray] = None
        self.step_count: int = 0

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Dimension of the state vector"""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Number of discrete actions (or dimensionality for continuous)"""
        pass

    @property
    def state(self) -> Optional[np.ndarray]:
        """
        Current state of the environment.
        Returns None if environment hasn't been reset.
        """
        if self.current_state is not None:
            return self.current_state.copy()
        return None

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            initial_state: Initial observation (numpy array)
        """
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action in environment.

        Args:
            action: Action to execute (integer for discrete actions)

        Returns:
            next_state: Next observation
            reward: Scalar reward
            done: Whether episode has terminated
            info: Additional information dictionary
        """
        pass

    @abstractmethod
    def get_state_components(self, state: Optional[np.ndarray] = None) -> Tuple:
        """
        Extract meaningful components from state for GVF cumulants.

        This method should return a tuple of interpretable state features
        that can be used to define predictive cumulants (e.g., position,
        velocity, angle, etc. for CartPole; grid properties for ARC).

        Args:
            state: State to decompose (uses current state if None)

        Returns:
            Tuple of state components (environment-specific)

        Example (CartPole):
            return (x, x_dot, theta, theta_dot)

        Example (ARC):
            return (grid_entropy, object_count, symmetry_score)
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up environment resources"""
        pass

    def get_action_space_info(self) -> Dict[str, Any]:
        """
        Get metadata about action space.

        Returns:
            Dictionary with keys:
            - 'type': 'discrete' or 'continuous'
            - 'n' or 'shape': Number of actions or continuous dimension
            - 'low', 'high': Bounds for continuous actions (optional)
        """
        return {
            'type': 'discrete',
            'n': self.action_dim
        }

    def get_observation_space_info(self) -> Dict[str, Any]:
        """
        Get metadata about observation space.

        Returns:
            Dictionary with keys:
            - 'type': 'vector' or 'grid' or 'image'
            - 'shape': Shape of observation
            - 'low', 'high': Bounds (optional)
        """
        return {
            'type': 'vector',
            'shape': (self.state_dim,)
        }

    def render(self, mode: str = 'human'):
        """
        Render environment (optional, for visualization).

        Args:
            mode: Render mode ('human', 'rgb_array', etc.)
        """
        pass
