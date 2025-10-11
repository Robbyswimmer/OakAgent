"""
Option Library: dynamic GVF-derived options for CartPole.
The canonical upright/centering/stabilize option classes remain available for
FC-STOMP or other discovery routines, but the library no longer seeds them by
default—options emerge through the developmental loop.
"""
import math
import numpy as np
from .option import Option

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


class OptionLibrary:
    """
    Library of all options
    Manages creation, execution, and lifecycle of options
    """

    def __init__(self, state_dim, action_dim, config, meta_config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.meta_template = meta_config.copy() if meta_config is not None else None

        # No pre-seeded options; FC-STOMP or other discovery mechanisms populate this.
        self.options = []  # index -> Option or None
        self.subtask_option_map = {}
        self.protected_options = set(getattr(self.config, 'OPTION_PROTECTED_IDS', []))

    def _create_core_options(self):
        """Create the 4 core GVF-derived options"""
        # o1: Upright Left (for positive theta)
        core_options = [
            UprightOption(
                option_id=0,
                name="upright_left",
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                theta_threshold=self.config.OPTION_THETA_THRESHOLD,
                preferred_direction=1,
                max_length=self.config.OPTION_MAX_LENGTH,
            ),
            UprightOption(
                option_id=1,
                name="upright_right",
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                theta_threshold=self.config.OPTION_THETA_THRESHOLD,
                preferred_direction=-1,
                max_length=self.config.OPTION_MAX_LENGTH,
            ),
            CenteringOption(
                option_id=2,
                name="centering",
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                x_threshold=self.config.OPTION_X_THRESHOLD,
                max_length=self.config.OPTION_MAX_LENGTH,
            ),
            StabilizeOption(
                option_id=3,
                name="stabilize",
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                velocity_threshold=self.config.OPTION_VELOCITY_THRESHOLD,
                max_length=self.config.OPTION_MAX_LENGTH,
            ),
        ]

        for option in core_options:
            option_id, _ = self.add_option(option)
            self.subtask_option_map[option.name] = option_id

    def get_option(self, option_id):
        """Get option by ID"""
        if 0 <= option_id < len(self.options):
            return self.options[option_id]
        return None

    def is_protected(self, option_id):
        """Return whether the option should be shielded from pruning."""
        return option_id in self.protected_options

    def get_all_options(self):
        """Get all options"""
        return {idx: opt for idx, opt in enumerate(self.options) if opt is not None}

    def get_num_options(self):
        """Get number of options"""
        return len(self.options)

    def add_option(self, option):
        """Dynamically add a new option (for FC-STOMP)"""
        option_id = option.option_id if option.option_id is not None and option.option_id >= 0 else None
        appended = False

        if option_id is None:
            # Reuse freed slot if available
            for idx, existing in enumerate(self.options):
                if existing is None:
                    option_id = idx
                    break
            if option_id is None:
                option_id = len(self.options)
                self.options.append(None)
                appended = True
        else:
            while option_id >= len(self.options):
                self.options.append(None)
                appended = True

        option.option_id = option_id
        self.options[option_id] = option
        self._configure_option_optimizers(option)
        return option_id, appended

    def _meta_config_for(self, enabled, base_lr):
        if self.meta_template is None or not enabled:
            return None
        cfg = self.meta_template.copy()
        if base_lr is not None and base_lr > 0:
            cfg['init_log_alpha'] = math.log(base_lr)
        cfg['disabled'] = False
        return cfg

    def _configure_option_optimizers(self, option):
        if option is None:
            return
        policy_lr = getattr(self.config, 'OPTION_POLICY_LR', option.policy_lr)
        value_lr = getattr(self.config, 'OPTION_VALUE_LR', option.value_lr)
        policy_meta = self._meta_config_for(
            getattr(self.config, 'OPTION_POLICY_META_ENABLED', True),
            policy_lr,
        )
        value_meta = self._meta_config_for(
            getattr(self.config, 'OPTION_VALUE_META_ENABLED', True),
            value_lr,
        )
        option.configure_optimizers(
            policy_lr=policy_lr,
            value_lr=value_lr,
            policy_meta_config=policy_meta,
            value_meta_config=value_meta,
        )

    def can_initiate(self, option_id, state):
        option = self.get_option(option_id)
        if option is None:
            return False
        try:
            return option.initiation(state)
        except Exception:
            return False

    def remove_option(self, option_id):
        """Remove an option (pruning)"""
        if 0 <= option_id < len(self.options):
            self.options[option_id] = None
        to_delete = [name for name, oid in self.subtask_option_map.items() if oid == option_id]
        for name in to_delete:
            del self.subtask_option_map[name]

    def get_option_ids(self):
        """Return list of active option IDs."""
        return [idx for idx, opt in enumerate(self.options) if opt is not None]

    def get_statistics(self):
        """Get statistics for all options"""
        stats = {}
        for option_id, option in enumerate(self.options):
            if option is None:
                continue
            stats[option_id] = option.get_statistics()
        return stats

    def execute_option(self, option_id, env):
        """Execute a specific option in environment"""
        option = self.get_option(option_id)
        if option is None:
            return [], False

        if env.state is None:
            return [], False

        if not option.initiation(env.state):
            return [], False

        return option.execute(env)

    def register_subtask_option(self, subtask_name, option_id):
        self.subtask_option_map[subtask_name] = option_id

    def has_subtask_option(self, subtask_name):
        return subtask_name in self.subtask_option_map

    def create_option_from_subtask(self, subtask, option_id=None):
        """Create an option tailored to an FC-STOMP subtask."""

        if self.has_subtask_option(subtask['name']):
            return None

        option = ConstructedOption(
            option_id=option_id if option_id is not None else -1,
            name=f"fc_{subtask['name']}",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            subtask=subtask,
            max_length=self.config.OPTION_MAX_LENGTH,
        )

        assigned_id, is_new_slot = self.add_option(option)
        self.register_subtask_option(subtask['name'], assigned_id)
        return assigned_id, is_new_slot


class ConstructedOption(Option):
    """Option generated from FC-STOMP subtask specifications."""

    def __init__(self, option_id, name, state_dim, action_dim, subtask, max_length=10):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.subtask = subtask
        self.feature_predictor = subtask.get('feature_predictor')
        self.goal_type = subtask.get('goal_type', 'minimize')
        self.target_value = subtask.get('target_value', 0.0)
        self.reward_fn = subtask.get('reward_fn', lambda state: 0.0)
        self.tolerance = subtask.get('tolerance', 0.01)

    def _feature_value(self, state):
        if self.feature_predictor is not None:
            return float(self.feature_predictor.predict(state))

        feature_name = self.subtask.get('feature_name', '')
        if feature_name == 'uprightness':
            return abs(state[2])
        if feature_name == 'centering':
            return abs(state[0])
        if feature_name == 'stability':
            return abs(state[1]) + abs(state[3])
        return 0.0

    def initiation(self, state):
        value = self._feature_value(state)
        if self.goal_type == 'minimize':
            return value > self.target_value + self.tolerance
        if self.goal_type == 'maximize':
            return value < self.target_value - self.tolerance
        return True

    def termination(self, state):
        value = self._feature_value(state)
        if self.goal_type == 'minimize':
            achieved = value <= self.target_value
        elif self.goal_type == 'maximize':
            achieved = value >= self.target_value
        else:
            achieved = False
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        return float(self.reward_fn(state))
