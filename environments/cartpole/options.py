"""
CartPole-Specific Option Templates
Temporal abstractions for CartPole domain
"""
import numpy as np
from options.option import Option


class UprightOption(Option):
    """
    Upright Option: Reduce |theta| to near 0
    Two variants: UprightLeft (for positive theta), UprightRight (for negative theta)
    """

    def __init__(self, option_id, name, state_dim, action_dim,
                 theta_threshold=0.05, preferred_direction=None, max_length=10):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.theta_threshold = theta_threshold
        self.preferred_direction = preferred_direction

    def termination(self, state):
        """Terminate when |theta| < threshold or max steps"""
        theta = state[2]  # theta is 3rd component
        achieved = abs(theta) < self.theta_threshold
        return achieved, achieved

    def initiation(self, state):
        if self.preferred_direction is None:
            return True

        theta = state[2]
        # Require the pole to lean in the designated direction beyond half the threshold
        if self.preferred_direction > 0:
            return theta > self.theta_threshold * 0.5
        else:
            return theta < -self.theta_threshold * 0.5

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: shaped to encourage reducing theta magnitude"""
        theta = state[2]
        # Exponential shaping: reward is higher when closer to goal
        distance = abs(theta) / self.theta_threshold
        reward = np.exp(-2.0 * distance) - 1.0  # in [-1, 0] range

        # Bonus for being very close to goal
        if abs(theta) < self.theta_threshold:
            reward += 1.0

        # Penalty for moving away from preferred direction
        if self.preferred_direction is not None:
            same_direction = self.preferred_direction * theta
            if same_direction > 0:
                reward -= 0.2 * min(same_direction, 1.0)

        return reward * 5.0  # scale for stronger signal


class CenteringOption(Option):
    """Centering Option: Reduce |x| to center the cart"""

    def __init__(self, option_id, name, state_dim, action_dim,
                 x_threshold=0.1, max_length=10):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.x_threshold = x_threshold

    def termination(self, state):
        """Terminate when |x| < threshold"""
        x = state[0]  # x is 1st component
        achieved = abs(x) < self.x_threshold
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: shaped to encourage centering"""
        x = state[0]
        distance = abs(x) / self.x_threshold
        reward = np.exp(-2.0 * distance) - 1.0

        # Bonus for being centered
        if abs(x) < self.x_threshold:
            reward += 1.0

        return reward * 5.0


class StabilizeOption(Option):
    """Stabilize Option: Reduce velocities (|theta_dot| + |x_dot|)"""

    def __init__(self, option_id, name, state_dim, action_dim,
                 velocity_threshold=0.5, max_length=10):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.velocity_threshold = velocity_threshold

    def termination(self, state):
        """Terminate when velocities are small"""
        x_dot = state[1]
        theta_dot = state[3]
        velocity_magnitude = abs(x_dot) + abs(theta_dot)
        achieved = velocity_magnitude < self.velocity_threshold
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: shaped to encourage stability"""
        x_dot = state[1]
        theta_dot = state[3]
        velocity_magnitude = abs(x_dot) + abs(theta_dot)
        distance = velocity_magnitude / self.velocity_threshold
        reward = np.exp(-2.0 * distance) - 1.0

        # Bonus for being stable
        if velocity_magnitude < self.velocity_threshold:
            reward += 1.0

        return reward * 5.0


class BalanceOption(Option):
    """
    Balance Option: Combined upright + centered
    (Alternative composite option)
    """

    def __init__(self, option_id, name, state_dim, action_dim,
                 theta_threshold=0.05, x_threshold=0.2, max_length=15):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.theta_threshold = theta_threshold
        self.x_threshold = x_threshold

    def termination(self, state):
        """Terminate when both upright and centered"""
        x = state[0]
        theta = state[2]
        achieved = (abs(theta) < self.theta_threshold and
                   abs(x) < self.x_threshold)
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: shaped combined goal"""
        x = state[0]
        theta = state[2]

        # Distance to goal in each dimension
        theta_dist = abs(theta) / self.theta_threshold
        x_dist = abs(x) / self.x_threshold

        # Combined exponential shaping
        reward = np.exp(-2.0 * (theta_dist + x_dist)) - 1.0

        # Bonus for achieving goal
        if abs(theta) < self.theta_threshold and abs(x) < self.x_threshold:
            reward += 2.0

        return reward * 5.0
