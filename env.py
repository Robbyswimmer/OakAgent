"""
CartPole Environment Wrapper
Provides state, action, reward interface for OaK agent
"""
import gymnasium as gym
import numpy as np

class CartPoleEnv:
    """
    Wrapper for CartPole-v1 environment
    State: (x, x_dot, theta, theta_dot)
    Actions: {0: Left, 1: Right}
    Reward: +1 per step until failure
    """

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_dim = 4  # (x, x_dot, theta, theta_dot)
        self.action_dim = 2  # Left, Right
        self.current_state = None
        self.step_count = 0

    def reset(self):
        """Reset environment and return initial state"""
        self.current_state, _ = self.env.reset()
        self.step_count = 0
        return self.current_state.copy()

    def step(self, action):
        """
        Execute action in environment
        Returns: (next_state, reward, done, info)
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.current_state = next_state
        self.step_count += 1

        return next_state.copy(), reward, done, info

    def get_state_components(self, state=None):
        """
        Extract individual state components for GVF cumulants
        Returns: (x, x_dot, theta, theta_dot)
        """
        if state is None:
            state = self.current_state
        x, x_dot, theta, theta_dot = state
        return x, x_dot, theta, theta_dot

    def close(self):
        """Close environment"""
        self.env.close()

    @property
    def state(self):
        """Current state"""
        return self.current_state.copy() if self.current_state is not None else None
