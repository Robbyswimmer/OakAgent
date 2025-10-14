"""
Option Library: Generic option management for OaK Framework
Dynamic option registry for any environment

Environment-specific option templates should be imported from their respective packages:
- CartPole: environments.cartpole.options (UprightOption, CenteringOption, etc.)
- ARC: environments.arc.options (FillRegionOption, SymmetrizeOption, etc.)
"""
import math
import numpy as np
from .option import Option


class OptionLibrary:
    """
    Library of all options
    Manages creation, execution, and lifecycle of options
    Environment-agnostic - works with any option subclass
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
