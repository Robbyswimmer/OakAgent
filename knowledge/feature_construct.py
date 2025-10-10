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
import math
from collections import deque

import numpy as np
import torch


def _safe_mean(values):
    """Return the arithmetic mean, handling empty sequences."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_variance(values):
    """Population variance that gracefully handles short sequences."""
    n = len(values)
    if n == 0:
        return 0.0
    mean_value = _safe_mean(values)
    return sum((v - mean_value) ** 2 for v in values) / n


def _safe_std(values):
    """Population standard deviation for a sequence."""
    variance = _safe_variance(values)
    return math.sqrt(variance)


def _diff(values):
    """Compute first differences for a sequence."""
    if len(values) < 2:
        return []
    return [values[i + 1] - values[i] for i in range(len(values) - 1)]


def _correlation(x_values, y_values):
    """Pearson correlation between two sequences."""
    if not x_values or not y_values:
        return 0.0

    n = min(len(x_values), len(y_values))
    x_values = x_values[:n]
    y_values = y_values[:n]

    mean_x = _safe_mean(x_values)
    mean_y = _safe_mean(y_values)

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    std_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    std_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))

    denom = std_x * std_y
    if denom == 0:
        return 0.0

    return max(min(cov / denom, 1.0), -1.0)

class FeatureConstructor:
    """
    Feature mining from GVF predictions
    Identifies promising features for option creation
    """

    def __init__(self, horde_gvfs, config, dyn_model=None):
        self.horde = horde_gvfs
        self.config = config
        self.dyn_model = dyn_model

        # Track discovered features
        self.discovered_features = []

        # History for analysis
        self.feature_history = deque(maxlen=1000)
        self.controllability_log = deque(maxlen=200)

    def mine_features(self, state_history=None, current_step=0):
        """
        Mine GVF prediction histories for promising features

        Returns:
            tuple(promising_features, feature_stats)
        """
        promising_features = []
        feature_stats = []

        # Prefer on-the-fly predictions from recent states if available
        invalid_counts = {}
        history_sample_size = 0

        if state_history:
            recent_states = list(state_history)[-self.config.GVF_BUFFER_SIZE :]
            history_sample_size = len(recent_states)
            histories = {}
            for gvf_name, gvf in self.horde.gvfs.items():
                preds = []
                invalid_count = 0
                for state in recent_states:
                    if state is None:
                        invalid_count += 1
                        continue
                    if isinstance(state, np.ndarray) and not np.all(np.isfinite(state)):
                        invalid_count += 1
                        continue
                    value = gvf.predict(state)

                    # DEBUG: Log first few predictions at FC-STOMP time
                    if len(preds) < 3:
                        cumulant = gvf.compute_cumulant(state)
                        print(f"  [FC-STOMP {gvf_name}] pred={value:.2f}, cumulant={cumulant:.3f}")

                    if not np.isfinite(value):
                        invalid_count += 1
                        continue
                    preds.append(value)
                histories[gvf_name] = preds
                invalid_counts[gvf_name] = invalid_count
        else:
            histories = self.horde.get_prediction_histories()

        if not histories:
            histories = self.horde.get_prediction_histories()

        for gvf_name, predictions in histories.items():
            if not predictions:
                fallback_predictions = self.horde.get_prediction_histories().get(
                    gvf_name, []
                )
                if fallback_predictions:
                    predictions = fallback_predictions
                    histories[gvf_name] = predictions

            min_history = getattr(self.config, 'FC_HISTORY_MIN_LENGTH', 100)
            snapshot_idx = len(feature_stats)

            # Criteria for promising features:
            count = len(predictions)
            variance = _safe_variance(predictions) if count > 0 else float('nan')
            variance_threshold = self.config.FC_FEATURE_VARIANCE_THRESHOLD
            relaxed_threshold = getattr(
                self.config, 'FC_FEATURE_INITIAL_VARIANCE', variance_threshold
            )
            relax_steps = getattr(self.config, 'FC_FEATURE_RELAX_STEPS', 0)
            if current_step < relax_steps:
                variance_threshold = max(variance_threshold, relaxed_threshold)
            is_stable = variance < variance_threshold

            # 2. Bottleneck detection (low values that can be controlled)
            mean_value = _safe_mean(predictions) if count > 0 else float('nan')
            is_bottleneck = mean_value < 0.3  # threshold for "small" values (normalized GVFs)

            # 3. High-value survival predictions
            is_high_value = gvf_name == 'survival' and mean_value > 0.6

            dropped = invalid_counts.get(gvf_name, 0)

            feature_snapshot = {
                'name': gvf_name,
                'mean': mean_value,
                'variance': variance,
                'controllability': 0.0,
                'variance_threshold': variance_threshold,
                'normalized_variance': variance / max(variance_threshold, 1e-6)
                if np.isfinite(variance)
                else float('inf'),
                'min_history': min_history,
                'count': count,
                'dropped': dropped,
                'history_size': history_sample_size,
            }
            if history_sample_size > 0 and dropped >= history_sample_size:
                feature_snapshot['note'] = 'all_recent_states_invalid'
            feature_stats.append(feature_snapshot)
            self.feature_history.append(feature_snapshot)

            if count < min_history:
                continue

            if is_stable or is_bottleneck or is_high_value:
                # Compute controllability heuristic
                # Stable features (low variance) should have HIGH controllability
                # Inverse relationship: 1 - normalized_variance
                normalized_variance = min(
                    variance / max(variance_threshold, 1e-6), 1.0
                ) if np.isfinite(variance) else 1.0
                controllability = max(0.0, 1.0 - normalized_variance)

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
                    'bottleneck': is_bottleneck,
                    'stats_index': snapshot_idx,
                }

                feature_snapshot['controllability'] = controllability
                promising_features.append(feature_info)

        return promising_features, feature_stats

    def analyze_feature_controllability(self, feature_name, state_history, action_history):
        """Estimate how strongly actions influence a GVF feature."""
        if len(state_history) < 50:
            return 0.0

        gvf = self.horde[feature_name]

        feature_values = np.array(
            [gvf.predict(state) for state in state_history], dtype=np.float64
        )
        if feature_values.size < 2 or not np.isfinite(feature_values).all():
            return 0.0

        actions = np.array(action_history[: len(feature_values)], dtype=np.float64)
        if actions.size < 2:
            return 0.0

        # Align transitions so each action corresponds to change in the feature
        feature_curr = feature_values[:-1]
        feature_next = feature_values[1:]
        actions_curr = actions[:-1]

        if feature_name == 'survival':
            delta = feature_next - feature_curr  # want to grow survival horizon
        else:
            delta = feature_curr - feature_next  # want to reduce magnitude

        valid_mask = np.isfinite(delta)
        delta = delta[valid_mask]
        actions_curr = actions_curr[valid_mask]

        if delta.size < 20:
            return 0.0

        action_effects = []
        for action in np.unique(actions_curr):
            mask = actions_curr == action
            count = int(np.sum(mask))
            if count < 5:
                continue
            mean_delta = float(np.mean(delta[mask]))
            action_effects.append((count, mean_delta))

        if not action_effects:
            return 0.0

        counts = np.array([c for c, _ in action_effects], dtype=np.float64)
        effects = np.array([e for _, e in action_effects], dtype=np.float64)

        best_effect = float(np.max(effects))
        worst_effect = float(np.min(effects)) if effects.size > 1 else 0.0
        contrast = max(0.0, best_effect - worst_effect)
        progress = max(0.0, best_effect) + 0.5 * contrast

        if progress <= 0.0:
            return 0.0

        coverage = float(np.clip(np.mean(counts) / max(len(feature_curr), 1), 0.05, 1.0))
        feature_scale = float(np.std(feature_curr) + np.std(delta) + 1e-6)

        score = (progress * coverage) / feature_scale
        score = float(np.tanh(2.5 * score))  # squash to [0, 1)
        return max(0.0, min(1.0, score))

    def compute_model_based_controllability(self, gvf, recent_states, horizon=4):
        """
        Model-based controllability via action contrast lookahead.

        For each state, simulate both actions (LEFT=0, RIGHT=1) for H steps,
        measure cumulative GVF predictions, compute contrast.

        Args:
            gvf: GVF to evaluate
            recent_states: list of recent states to sample from
            horizon: lookahead steps (default 2-3 is good)

        Returns:
            float: mean action contrast Δ ∈ [0, 1+]
        """
        if self.dyn_model is None or not recent_states:
            return 0.0

        # Sample up to 200 states for efficiency
        n_samples = min(200, len(recent_states))
        if n_samples < 10:
            return 0.0

        sample_indices = np.random.choice(len(recent_states), n_samples, replace=False)
        sampled_states = [recent_states[i] for i in sample_indices]

        contrasts = []

        for state in sampled_states:
            if state is None or not np.all(np.isfinite(state)):
                continue

            # Simulate both actions
            action_scores = []

            for action in [0, 1]:  # LEFT, RIGHT
                s_sim = np.array(state, dtype=np.float32, copy=True)
                cumulative_gvf = 0.0
                discount = 1.0

                for h in range(horizon):
                    # One-step model prediction
                    s_next, r_pred = self.dyn_model.predict(s_sim, action)

                    # Extract next state from tensor
                    if isinstance(s_next, torch.Tensor):
                        s_next = s_next.squeeze(0).cpu().numpy()

                    # GVF prediction at next state
                    gvf_value = gvf.predict(s_next)

                    if not np.isfinite(gvf_value):
                        break

                    cumulative_gvf += discount * gvf_value
                    discount *= 0.9  # decay for multi-step
                    s_sim = s_next

                    # Early stop if terminal-looking state
                    if abs(s_sim[0]) > 2.4 or abs(s_sim[2]) > 0.2095:
                        break

                action_scores.append(cumulative_gvf)

            # Contrast between actions
            if len(action_scores) == 2:
                contrast = abs(action_scores[0] - action_scores[1])
                if np.isfinite(contrast):
                    contrasts.append(np.clip(contrast, 0.0, 1.5))

        if not contrasts:
            return 0.0

        contrasts = np.array(contrasts, dtype=np.float64)
        score = float(np.tanh(contrasts.mean()))
        score = max(0.0, min(1.0, score))
        self.controllability_log.append(score)
        return score


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
                'reward_fn': lambda state: -abs(state[2]),
                'feature_predictor': feature_info.get('gvf'),
                'tolerance': 0.01,
            }

        elif feature_name == 'centering':
            # Goal: minimize |x|
            subtask = {
                'name': f'minimize_{feature_name}',
                'feature_name': feature_name,
                'goal_type': 'minimize',
                'target_value': self.config.OPTION_X_THRESHOLD,
                'reward_fn': lambda state: -abs(state[0]),
                'feature_predictor': feature_info.get('gvf'),
                'tolerance': 0.02,
            }

        elif feature_name == 'stability':
            # Goal: minimize velocities
            subtask = {
                'name': f'minimize_{feature_name}',
                'feature_name': feature_name,
                'goal_type': 'minimize',
                'target_value': self.config.OPTION_VELOCITY_THRESHOLD,
                'reward_fn': lambda state: -(abs(state[1]) + abs(state[3])),
                'feature_predictor': feature_info.get('gvf'),
                'tolerance': 0.05,
            }

        elif feature_name == 'survival':
            # Goal: maximize survival time
            subtask = {
                'name': f'maximize_{feature_name}',
                'feature_name': feature_name,
                'goal_type': 'maximize',
                'target_value': self.config.FC_SURVIVAL_TARGET,
                'reward_fn': lambda state: 1.0,
                'feature_predictor': feature_info.get('gvf'),
                'tolerance': 5.0,
            }

        else:
            # Generic subtask
            subtask = {
                'name': f'control_{feature_name}',
                'feature_name': feature_name,
                'goal_type': 'minimize',
                'target_value': 0.1,
                'reward_fn': lambda state: -feature_info['gvf'].predict(state),
                'feature_predictor': feature_info.get('gvf'),
                'tolerance': 0.05,
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

    def evaluate_options(self, current_step, recent_usage=None):
        """
        Evaluate all options and identify candidates for pruning

        Returns:
            list of option_ids to prune
        """
        to_prune = []

        stats = self.option_library.get_statistics()
        recent_usage = recent_usage or {}
        recent_min_starts = getattr(self.config, 'FC_OPTION_PRUNE_RECENT_STARTS', 5)

        min_executions = getattr(self.config, 'FC_OPTION_MIN_EXECUTIONS', 10)
        min_age = getattr(self.config, 'FC_OPTION_PRUNE_MIN_AGE_STEPS', 0)

        for option_id, option_stats in stats.items():
            option = self.option_library.get_option(option_id)

            if option is not None and hasattr(option, 'creation_step'):
                option_age = current_step - option.creation_step
                if option_age < min_age:
                    continue

            if option_stats['executions'] < max(min_executions, 10):
                # Not enough data yet
                continue

            # Pruning criteria:
            # 1. Low success rate
            if option_stats['success_rate'] < self.config.FC_OPTION_SUCCESS_THRESHOLD:
                to_prune.append(option_id)
                continue

            # 2. Recent usage underperforms (skip protected options)
            if not self.option_library.is_protected(option_id):
                usage_stats = recent_usage.get(option_id)
                if usage_stats:
                    if (
                        usage_stats['starts'] >= recent_min_starts
                        and usage_stats['success_rate'] < self.config.FC_OPTION_SUCCESS_THRESHOLD
                    ):
                        to_prune.append(option_id)
                        continue
                elif recent_usage is not None and recent_min_starts > 0:
                    to_prune.append(option_id)

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
                 q_option, config, dyn_model=None):
        self.horde = horde_gvfs
        self.option_library = option_library
        self.option_models = option_models
        self.q_option = q_option
        self.config = config
        self.dyn_model = dyn_model

        # Sub-components
        self.feature_constructor = FeatureConstructor(horde_gvfs, config, dyn_model)
        self.subtask_former = SubtaskFormation(config)
        self.pruner = OptionPruner(option_library, config)

        # Track FC-STOMP events
        self.fc_stomp_history = []
        self.last_run_step = 0
        self.subtask_to_option = {}
        self.feature_last_spawn_step = {}

    def run_fc_stomp_cycle(
        self,
        current_step,
        state_history=None,
        action_history=None,
        recent_option_usage=None,
    ):
        """
        Execute one FC-STOMP cycle

        Args:
            current_step: current training step
            state_history: recent state history (for controllability analysis)
            action_history: recent action history
            recent_option_usage: aggregated option usage metrics over recent episodes

        Returns:
            dict with cycle results
        """
        results = {
            'step': current_step,
            'features_mined': 0,
            'subtasks_formed': 0,
            'options_created': 0,
            'options_pruned': 0,
            'feature_stats': []
        }

        # F: Feature Construction
        promising_features, feature_stats = self.feature_constructor.mine_features(
            state_history, current_step=current_step
        )
        results['features_mined'] = len(promising_features)
        results['feature_stats'] = feature_stats

        # C: Subtask Formation
        optionless = not self.option_library.get_option_ids()
        min_controllability = self.config.FC_MIN_CONTROLLABILITY
        if optionless:
            min_controllability = getattr(
                self.config,
                'FC_MIN_CONTROLLABILITY_BOOTSTRAP',
                max(0.0, self.config.FC_MIN_CONTROLLABILITY * 0.5),
            )

        considered_features = {fi['name'] for fi in promising_features}

        for feature_info in promising_features:
            # Check controllability using model-based lookahead if available
            if state_history and self.dyn_model is not None:
                gvf = feature_info['gvf']
                recent_states = list(state_history)[-self.config.GVF_BUFFER_SIZE :]
                controllability_model = self.feature_constructor.compute_model_based_controllability(
                    gvf,
                    recent_states,
                    horizon=getattr(self.config, 'FC_CONTROLLABILITY_H', 2),
                )
                base_controllability = feature_info.get('controllability', 0.0)
                feature_info['controllability'] = max(base_controllability, controllability_model)
            elif state_history and action_history:
                # Fallback to historical correlation method
                controllability = self.feature_constructor.analyze_feature_controllability(
                    feature_info['name'], state_history, action_history
                )
                feature_info['controllability'] = controllability

            stats_idx = feature_info.get('stats_index')
            if stats_idx is not None and 0 <= stats_idx < len(results['feature_stats']):
                results['feature_stats'][stats_idx]['controllability'] = feature_info.get(
                    'controllability', 0.0
                )
            feature_info.pop('stats_index', None)

            # Only create subtask if controllable enough (relaxed if no options yet)
            if feature_info['controllability'] >= min_controllability:
                subtask = self.subtask_former.create_subtask(feature_info)
                results['subtasks_formed'] += 1

                # S: ensure an option exists for this subtask
                created = self._ensure_option_for_subtask(subtask, current_step)
                if created:
                    results['options_created'] += 1

        # Consider model-based controllability for features that were not initially marked promising
        if state_history and self.dyn_model is not None:
            recent_states = list(state_history)[-self.config.GVF_BUFFER_SIZE :]
            if recent_states:
                delta_threshold = getattr(
                    self.config,
                    'FC_MODEL_CONTROLLABILITY_MIN',
                    min_controllability,
                )
                horizon = getattr(self.config, 'FC_CONTROLLABILITY_H', 2)
                for name, gvf in self.horde.gvfs.items():
                    if name in considered_features:
                        continue
                    model_ctrl = self.feature_constructor.compute_model_based_controllability(
                        gvf,
                        recent_states,
                        horizon=horizon,
                    )
                    if model_ctrl < delta_threshold:
                        continue

                    stats_idx = next(
                        (idx for idx, snapshot in enumerate(results['feature_stats']) if snapshot['name'] == name),
                        None,
                    )

                    feature_info = {
                        'name': name,
                        'gvf': gvf,
                        'controllability': model_ctrl,
                    }

                    if stats_idx is not None:
                        results['feature_stats'][stats_idx]['controllability'] = model_ctrl

                    if model_ctrl >= min_controllability:
                        subtask = self.subtask_former.create_subtask(feature_info)
                        results['subtasks_formed'] += 1

                        created = self._ensure_option_for_subtask(subtask, current_step)
                        if created:
                            results['options_created'] += 1
                            considered_features.add(name)

        if optionless and results['options_created'] == 0:
            print("[FC-STOMP] Scouting phase: no controllable features yet; retrying later.")

        # Prune underperforming options
        options_to_prune = self.pruner.evaluate_options(current_step, recent_option_usage)
        for option_id in options_to_prune:
            # Only process options that are not protected by configuration.
            if not self.option_library.is_protected(option_id):
                if hasattr(self.option_models, 'remove_option'):
                    self.option_models.remove_option(option_id)
                if hasattr(self.q_option, 'reset_option'):
                    self.q_option.reset_option(option_id)
                self.pruner.prune_option(option_id)
                results['options_pruned'] += 1

        # Log event
        self.fc_stomp_history.append(results)
        self.last_run_step = current_step

        return results

    def _ensure_option_for_subtask(self, subtask, current_step):
        name = subtask['name']
        if name in self.subtask_to_option:
            return False

        feature_name = subtask.get('feature_name')
        cooldown = getattr(self.config, 'FC_FEATURE_SPAWN_COOLDOWN', 0)
        if feature_name is not None:
            last_spawn = self.feature_last_spawn_step.get(feature_name, -np.inf)
            if current_step - last_spawn < cooldown:
                return False

        if (
            self.option_library is None
            or not hasattr(self.option_library, 'create_option_from_subtask')
            or self.option_models is None
            or not hasattr(self.option_models, 'add_option')
            or self.q_option is None
            or not hasattr(self.q_option, 'add_option')
        ):
            self.subtask_to_option[name] = None
            return False

        create_result = self.option_library.create_option_from_subtask(subtask)
        if create_result is None:
            existing = None
            if hasattr(self.option_library, 'subtask_option_map'):
                existing = self.option_library.subtask_option_map.get(name)
            self.subtask_to_option[name] = existing
            return False

        option_id, is_new_slot = create_result

        # Reinitialize or add supporting models/q-values
        self.option_models.add_option(option_id)
        if hasattr(self.q_option, 'reset_option') and not is_new_slot:
            self.q_option.reset_option(option_id)
        else:
            self.q_option.add_option()

        self.subtask_to_option[name] = option_id
        if feature_name is not None:
            self.feature_last_spawn_step[feature_name] = current_step
        option = self.option_library.get_option(option_id)
        if option is not None and hasattr(option, 'creation_step'):
            option.creation_step = current_step
        return True

    def should_run(self, current_step):
        """Check if FC-STOMP should run at this step"""
        if current_step - self.last_run_step >= self.config.FC_STOMP_FREQ:
            return True
        return False

    def get_history(self):
        """Get FC-STOMP execution history"""
        return self.fc_stomp_history
