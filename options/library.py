"""
Option Library: 4 Core Options for CartPole
Each option targets a specific GVF-derived feature:
- o1/o2: Uprightness (reduce |theta|)
- o3: Centering (reduce |x|)
- o4: Stabilize (reduce velocities)
"""
import numpy as np
from .option import Option

class UprightOption(Option):
    """
    Upright Option: Reduce |theta| to near 0
    Two variants: UprightLeft (for positive theta), UprightRight (for negative theta)
    """

    def __init__(self, option_id, name, state_dim, action_dim,
                 theta_threshold=0.05, max_length=10):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.theta_threshold = theta_threshold

    def termination(self, state):
        """Terminate when |theta| < threshold or max steps"""
        theta = state[2]  # theta is 3rd component
        achieved = abs(theta) < self.theta_threshold
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: negative angle magnitude (want to minimize)"""
        theta = state[2]
        return -abs(theta)


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
        """Pseudo-reward: negative position magnitude"""
        x = state[0]
        return -abs(x)


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
        """Pseudo-reward: negative velocity magnitude"""
        x_dot = state[1]
        theta_dot = state[3]
        return -(abs(x_dot) + abs(theta_dot))


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
        """Pseudo-reward: combined distance from goal"""
        x = state[0]
        theta = state[2]
        return -(abs(theta) + abs(x))


class OptionLibrary:
    """
    Library of all options
    Manages creation, execution, and lifecycle of options
    """

    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config

        # Create initial 4 core options
        self.options = {}
        self._create_core_options()

        # Track next option ID for dynamic creation
        self.next_option_id = len(self.options)

    def _create_core_options(self):
        """Create the 4 core GVF-derived options"""
        # o1: Upright Left (for positive theta)
        self.options[0] = UprightOption(
            option_id=0,
            name="upright_left",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            theta_threshold=self.config.OPTION_THETA_THRESHOLD,
            max_length=self.config.OPTION_MAX_LENGTH
        )

        # o2: Upright Right (for negative theta)
        self.options[1] = UprightOption(
            option_id=1,
            name="upright_right",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            theta_threshold=self.config.OPTION_THETA_THRESHOLD,
            max_length=self.config.OPTION_MAX_LENGTH
        )

        # o3: Centering
        self.options[2] = CenteringOption(
            option_id=2,
            name="centering",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            x_threshold=self.config.OPTION_X_THRESHOLD,
            max_length=self.config.OPTION_MAX_LENGTH
        )

        # o4: Stabilize
        self.options[3] = StabilizeOption(
            option_id=3,
            name="stabilize",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            velocity_threshold=self.config.OPTION_VELOCITY_THRESHOLD,
            max_length=self.config.OPTION_MAX_LENGTH
        )

    def get_option(self, option_id):
        """Get option by ID"""
        return self.options.get(option_id)

    def get_all_options(self):
        """Get all options"""
        return self.options.copy()

    def get_num_options(self):
        """Get number of options"""
        return len(self.options)

    def add_option(self, option):
        """Dynamically add a new option (for FC-STOMP)"""
        self.options[self.next_option_id] = option
        self.next_option_id += 1
        return self.next_option_id - 1

    def remove_option(self, option_id):
        """Remove an option (pruning)"""
        if option_id in self.options:
            del self.options[option_id]

    def get_statistics(self):
        """Get statistics for all options"""
        stats = {}
        for option_id, option in self.options.items():
            stats[option_id] = option.get_statistics()
        return stats

    def execute_option(self, option_id, env):
        """Execute a specific option in environment"""
        option = self.get_option(option_id)
        if option is None:
            raise ValueError(f"Option {option_id} does not exist")

        if not option.initiation(env.state):
            raise ValueError(f"Option {option_id} cannot be initiated from current state")

        return option.execute(env)
