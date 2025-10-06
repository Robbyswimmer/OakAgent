"""
FC-STOMP: Feature Construction → Subtask → Option
The developmental loop that enables hierarchical growth (OaK principle #8)

Stages:
F: Feature Construction - mine GVF predictions for salient patterns
C: Subtask Formation - identify controllable aspects
S: Subtask → Option - create option to achieve subtask
T: Train Option & Model - learn option policy and SMDP model
O: Option-Model Integration - integrate into planner
M: Meta-Learning - adapt step-sizes
P: Planning & Acting - use in decision-making
"""
import numpy as np
from collections import deque

class FeatureConstructor:
    """
    Feature mining from GVF predictions
    Identifies promising features for option creation
    """

    def __init__(self, horde_gvfs, config):
        self.horde = horde_gvfs
        self.config = config

        # Track discovered features
        self.discovered_features = []

        # History for analysis
        self.feature_history = deque(maxlen=1000)

    def mine_features(self):
        """
        Mine GVF prediction histories for promising features

        Returns:
            list of (feature_name, feature_fn, controllability_score)
        """
        promising_features = []

        # Get prediction histories from all GVFs
        histories = self.horde.get_prediction_histories()

        for gvf_name, predictions in histories.items():
            if len(predictions) < 100:
                continue

            predictions_array = np.array(predictions)

            # Criteria for promising features:

            # 1. Low variance (stable prediction)
            variance = np.var(predictions_array)
            is_stable = variance < self.config.FC_FEATURE_VARIANCE_THRESHOLD

            # 2. Bottleneck detection (low values that can be controlled)
            mean_value = np.mean(predictions_array)
            is_bottleneck = mean_value < 0.5  # threshold for "small" values

            # 3. High-value survival predictions
            is_high_value = gvf_name == 'survival' and mean_value > 100

            if is_stable or is_bottleneck or is_high_value:
                # Compute controllability heuristic
                # Stable features (low variance) should have HIGH controllability
                # Inverse relationship: 1 - normalized_variance
                normalized_variance = min(variance / 0.5, 1.0)  # cap at 0.5
                controllability = 1.0 - normalized_variance  # Stable = high controllability

                # Alternative: if bottleneck or high-value, boost controllability
                if is_bottleneck or is_high_value:
                    controllability = max(controllability, 0.8)

                feature_info = {
                    'name': gvf_name,
                    'gvf': self.horde[gvf_name],
                    'mean': mean_value,
                    'variance': variance,
                    'controllability': controllability,
                    'stable': is_stable,
                    'bottleneck': is_bottleneck
                }

                promising_features.append(feature_info)

        return promising_features

    def analyze_feature_controllability(self, feature_name, state_history, action_history):
        """
        Analyze whether a feature can be controlled by actions
        Computes correlation between actions and feature changes
        """
        if len(state_history) < 50:
            return 0.0

        # Get feature values from states
        feature_values = []
        gvf = self.horde[feature_name]

        for state in state_history:
            value = gvf.predict(state)
            feature_values.append(value)

        # Compute correlation with actions (simple measure)
        if len(action_history) < len(feature_values):
            action_history = action_history + [0] * (len(feature_values) - len(action_history))

        # Measure how much feature changes when actions change
        feature_diffs = np.diff(feature_values[:len(action_history)-1])
        action_changes = np.diff(action_history[:len(feature_values)-1])

        if len(feature_diffs) == 0 or np.std(action_changes) == 0:
            return 0.0

        # Correlation
        correlation = np.abs(np.corrcoef(feature_diffs, action_changes)[0, 1])
        return correlation if not np.isnan(correlation) else 0.0


class SubtaskFormation:
    """
    Converts promising features into subtask specifications
    Subtask = goal state defined by feature constraints
    """

    def __init__(self, config):
        self.config = config
        self.subtasks = []

    def create_subtask(self, feature_info):
        """
        Create a subtask from a promising feature

        Args:
            feature_info: dict with feature metadata

        Returns:
            subtask specification
        """
        feature_name = feature_info['name']

        # Define subtask based on feature type
        if feature_name == 'uprightness':
            # Goal: minimize |theta|
            subtask = {
                'name': f'minimize_{feature_name}',
                'feature_name': feature_name,
                'goal_type': 'minimize',
                'target_value': self.config.OPTION_THETA_THRESHOLD,
                'reward_fn': lambda state: -abs(state[2])  # -|theta|
            }

        elif feature_name == 'centering':
            # Goal: minimize |x|
            subtask = {
                'name': f'minimize_{feature_name}',
                'feature_name': feature_name,
                'goal_type': 'minimize',
                'target_value': self.config.OPTION_X_THRESHOLD,
                'reward_fn': lambda state: -abs(state[0])  # -|x|
            }

        elif feature_name == 'stability':
            # Goal: minimize velocities
            subtask = {
                'name': f'minimize_{feature_name}',
                'feature_name': feature_name,
                'goal_type': 'minimize',
                'target_value': self.config.OPTION_VELOCITY_THRESHOLD,
                'reward_fn': lambda state: -(abs(state[1]) + abs(state[3]))
            }

        elif feature_name == 'survival':
            # Goal: maximize survival time
            subtask = {
                'name': f'maximize_{feature_name}',
                'feature_name': feature_name,
                'goal_type': 'maximize',
                'target_value': float('inf'),
                'reward_fn': lambda state: 1.0  # constant positive reward
            }

        else:
            # Generic subtask
            subtask = {
                'name': f'control_{feature_name}',
                'feature_name': feature_name,
                'goal_type': 'minimize',
                'target_value': 0.1,
                'reward_fn': lambda state: -feature_info['gvf'].predict(state)
            }

        self.subtasks.append(subtask)
        return subtask


class OptionPruner:
    """
    Manages option lifecycle: creation and pruning
    Removes options that don't improve performance
    """

    def __init__(self, option_library, config):
        self.option_library = option_library
        self.config = config

        # Track option performance
        self.option_performance = {}

    def evaluate_options(self):
        """
        Evaluate all options and identify candidates for pruning

        Returns:
            list of option_ids to prune
        """
        to_prune = []

        stats = self.option_library.get_statistics()

        for option_id, option_stats in stats.items():
            if option_stats['executions'] < 10:
                # Not enough data yet
                continue

            # Pruning criteria:
            # 1. Low success rate
            if option_stats['success_rate'] < self.config.FC_OPTION_SUCCESS_THRESHOLD:
                to_prune.append(option_id)

            # 2. Never used recently (could add usage tracking)

        return to_prune

    def prune_option(self, option_id):
        """Remove an option from the library"""
        self.option_library.remove_option(option_id)
        if option_id in self.option_performance:
            del self.option_performance[option_id]


class FCSTOMPManager:
    """
    Main FC-STOMP manager
    Orchestrates the full feature → subtask → option pipeline
    """

    def __init__(self, horde_gvfs, option_library, option_models,
                 q_option, config):
        self.horde = horde_gvfs
        self.option_library = option_library
        self.option_models = option_models
        self.q_option = q_option
        self.config = config

        # Sub-components
        self.feature_constructor = FeatureConstructor(horde_gvfs, config)
        self.subtask_former = SubtaskFormation(config)
        self.pruner = OptionPruner(option_library, config)

        # Track FC-STOMP events
        self.fc_stomp_history = []
        self.last_run_step = 0

    def run_fc_stomp_cycle(self, current_step, state_history=None, action_history=None):
        """
        Execute one FC-STOMP cycle

        Args:
            current_step: current training step
            state_history: recent state history (for controllability analysis)
            action_history: recent action history

        Returns:
            dict with cycle results
        """
        results = {
            'step': current_step,
            'features_mined': 0,
            'subtasks_formed': 0,
            'options_created': 0,
            'options_pruned': 0
        }

        # F: Feature Construction
        promising_features = self.feature_constructor.mine_features()
        results['features_mined'] = len(promising_features)

        # C: Subtask Formation
        for feature_info in promising_features:
            # Check controllability if history available
            if state_history and action_history:
                controllability = self.feature_constructor.analyze_feature_controllability(
                    feature_info['name'], state_history, action_history
                )
                feature_info['controllability'] = controllability

            # Only create subtask if controllable enough
            if feature_info['controllability'] >= self.config.FC_MIN_CONTROLLABILITY:
                subtask = self.subtask_former.create_subtask(feature_info)
                results['subtasks_formed'] += 1

                # S: Check if we need a new option for this subtask
                # (For now, we start with 4 core options, so skip dynamic creation)
                # In full implementation, would create new option here

        # Prune underperforming options
        options_to_prune = self.pruner.evaluate_options()
        for option_id in options_to_prune:
            # Only prune if not a core option (keep 0-3)
            if option_id >= 4:
                self.pruner.prune_option(option_id)
                results['options_pruned'] += 1

        # Log event
        self.fc_stomp_history.append(results)
        self.last_run_step = current_step

        return results

    def should_run(self, current_step):
        """Check if FC-STOMP should run at this step"""
        if current_step - self.last_run_step >= self.config.FC_STOMP_FREQ:
            return True
        return False

    def get_history(self):
        """Get FC-STOMP execution history"""
        return self.fc_stomp_history
