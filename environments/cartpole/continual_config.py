"""
Continual Learning Configuration
Extends base config with regime switching for non-stationary RL experiments
"""
from config import Config
from continual_env import create_regime_schedule


class ContinualConfig(Config):
    """Configuration for continual learning experiments"""

    # ========== Continual Learning Settings ==========

    # Regime schedule: List of (regime_name, start_episode, end_episode)
    # Default: 5 regimes × 200 episodes each = 1000 total episodes (max)
    # Note: May switch earlier if regime is "solved"
    REGIME_SCHEDULE = create_regime_schedule(
        episodes_per_regime=200,
        regimes=['R1_base', 'R2_heavy', 'R3_long', 'R4_friction', 'R5_gravity']
    )

    # Total episodes spans all regimes (upper bound)
    NUM_EPISODES = 1000

    # Early termination: switch to next regime when solved
    EARLY_REGIME_SWITCH = True  # If True, switch when solved (before episode limit)
    REGIME_SOLVED_THRESHOLD = 475.0  # Average return over window to consider "solved"
    REGIME_SOLVED_WINDOW = 20  # Episodes to average over for solved check
    MIN_EPISODES_PER_REGIME = 50  # Minimum episodes before checking solved condition

    # Continual learning evaluation
    CONTINUAL_EVAL_FREQ = 10  # Evaluate every N episodes
    CONTINUAL_EVAL_EPISODES = 20  # Fewer episodes per eval (more frequent)
    EVAL_AT_REGIME_TRANSITIONS = True  # Force eval right before/after regime switch

    # Adaptation metrics
    TRACK_REGIME_PERFORMANCE = True
    ADAPTATION_WINDOW = 20  # Episodes to measure adaptation speed
    PERFORMANCE_RECOVERY_THRESHOLD = 0.9  # 90% of pre-transition performance

    # Logging
    LOG_REGIME_TRANSITIONS = True
    LOG_ADAPTATION_METRICS = True

    # ========== Modified Training Settings for Continual Learning ==========

    # Use base epsilon decay (agent needs to solve each regime)
    EPSILON_DECAY = 0.998   # Same as base config
    EPSILON_END = 0.01      # Same as base config

    # FC-STOMP frequency (same as base config)
    FC_STOMP_FREQ = 500  # Environment steps between FC-STOMP

    # Evaluation settings
    EVAL_FREQ = 10  # Evaluate every 10 episodes
    EVAL_EPISODES = 20  # Fewer episodes per eval for speed

    # ========== Ablation Flags for Continual Learning ==========

    # Compare different continual learning strategies
    ABLATION_RESET_DYNAMICS_AT_TRANSITION = False  # If True, reset dynamics model at transitions
    ABLATION_RESET_OPTIONS_AT_TRANSITION = False  # If True, prune all options at transitions
    ABLATION_FREEZE_GVFS_AFTER_R1 = False  # If True, stop GVF learning after R1

    # ========== Continual Learning Adaptations ==========

    # Epsilon boost on regime shifts (Fix #1)
    EPSILON_BOOST_ON_SHIFT = True  # Boost epsilon when regime changes
    EPSILON_BOOST_VALUE = 0.5  # Target epsilon after regime shift

    # Model error gating for planning (Fix #3)
    PLANNING_ERROR_THRESHOLD = 0.5  # Disable planning if model MAE > this

    # Distribution shift detection (Fix #4)
    SHIFT_DETECTION_ENABLED = True  # Enable automatic shift detection
    SHIFT_DETECTION_SPIKE_THRESHOLD = 2.0  # Model error must spike by this factor
    SHIFT_DETECTION_LOOKBACK = 20  # Baseline window for error comparison

    # Regime-aware option management (Fix #6)
    REGIME_AWARE_PRUNING = True  # Aggressive option pruning after shift
    AGGRESSIVE_PRUNING_WINDOW = 1000  # Steps after shift to use aggressive thresholds

    @classmethod
    def get_current_regime(cls, episode: int) -> str:
        """
        Get the regime for a given episode number.

        Args:
            episode: Episode number

        Returns:
            Regime name (e.g., 'R1_base')
        """
        for regime_name, start_ep, end_ep in cls.REGIME_SCHEDULE:
            if start_ep <= episode < end_ep:
                return regime_name
        # Return last regime if past schedule
        return cls.REGIME_SCHEDULE[-1][0]

    @classmethod
    def is_regime_transition(cls, episode: int) -> bool:
        """
        Check if this episode is a regime transition point.

        Args:
            episode: Episode number

        Returns:
            True if this is the first episode of a new regime
        """
        for regime_name, start_ep, end_ep in cls.REGIME_SCHEDULE:
            if episode == start_ep and episode > 0:
                return True
        return False

    @classmethod
    def get_regime_boundaries(cls) -> list:
        """
        Get list of episode numbers where regimes change.

        Returns:
            List of episode numbers marking regime boundaries
        """
        boundaries = []
        for regime_name, start_ep, end_ep in cls.REGIME_SCHEDULE:
            if start_ep > 0:
                boundaries.append(start_ep)
        return boundaries


# Alternative schedules for different experiments

class FastAdaptationConfig(ContinualConfig):
    """Shorter regimes for faster adaptation testing"""
    REGIME_SCHEDULE = create_regime_schedule(
        episodes_per_regime=50,  # Only 50 episodes per regime
        regimes=['R1_base', 'R2_heavy', 'R3_long', 'R4_friction', 'R5_gravity']
    )
    NUM_EPISODES = 250
    ADAPTATION_WINDOW = 10


class TwoRegimeConfig(ContinualConfig):
    """Simplified two-regime setup for initial testing"""
    REGIME_SCHEDULE = create_regime_schedule(
        episodes_per_regime=100,
        regimes=['R1_base', 'R2_heavy']
    )
    NUM_EPISODES = 200


class BackAndForthConfig(ContinualConfig):
    """Test backward transfer: R1 → R2 → R1"""
    REGIME_SCHEDULE = [
        ('R1_base', 0, 100),
        ('R2_heavy', 100, 200),
        ('R1_base', 200, 300),  # Back to R1 - test catastrophic forgetting
    ]
    NUM_EPISODES = 300


class GradualShiftConfig(ContinualConfig):
    """Gradual parameter interpolation (for future work)"""
    # TODO: Implement smooth transitions between regimes
    REGIME_SCHEDULE = create_regime_schedule(
        episodes_per_regime=100,
        regimes=['R1_base', 'R2_heavy', 'R3_long']
    )
    NUM_EPISODES = 300
    GRADUAL_TRANSITION_EPISODES = 10  # Linearly interpolate params over 10 episodes
