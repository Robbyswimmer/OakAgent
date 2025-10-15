"""
Continual Learning Metrics
Track and analyze performance across regime transitions
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json


class ContinualMetrics:
    """
    Track continual learning performance across regime transitions.

    Metrics computed:
    - Per-regime average return
    - Adaptation time (episodes to recover performance)
    - Forward transfer (initial performance on new regime)
    - Backward transfer (performance degradation on old regimes)
    - Forgetting rate
    """

    def __init__(self, regime_schedule: List[Tuple[str, int, int]]):
        """
        Initialize metrics tracker.

        Args:
            regime_schedule: List of (regime_name, start_episode, end_episode)
        """
        self.regime_schedule = regime_schedule
        self.regime_boundaries = [start for _, start, _ in regime_schedule if start > 0]

        # Episode-level tracking
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_regimes: List[str] = []

        # Regime-level tracking
        self.regime_returns: Dict[str, List[float]] = defaultdict(list)
        self.regime_eval_returns: Dict[str, List[float]] = defaultdict(list)

        # Transition metrics
        self.pre_transition_performance: Dict[int, float] = {}  # boundary_ep -> avg_return
        self.post_transition_performance: Dict[int, float] = {}
        self.adaptation_times: Dict[int, int] = {}  # boundary_ep -> episodes_to_adapt

        # GVF and option metrics across regimes
        self.gvf_errors: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.option_success_rates: Dict[str, List[Dict[int, float]]] = defaultdict(list)

    def log_episode(self, episode: int, episode_return: float, episode_length: int,
                    regime: str, eval_return: Optional[float] = None,
                    gvf_errors: Optional[Dict[str, float]] = None,
                    option_stats: Optional[Dict[int, Dict]] = None):
        """
        Log episode results.

        Args:
            episode: Episode number
            episode_return: Return for this episode
            episode_length: Steps in this episode
            regime: Current regime name
            eval_return: Evaluation return (if evaluated this episode)
            gvf_errors: Dict of GVF name -> error
            option_stats: Dict of option_id -> stats dict
        """
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)
        self.episode_regimes.append(regime)

        self.regime_returns[regime].append(episode_return)

        if eval_return is not None:
            self.regime_eval_returns[regime].append(eval_return)

        if gvf_errors is not None:
            self.gvf_errors[regime].append(gvf_errors)

        if option_stats is not None:
            success_rates = {
                opt_id: stats.get('success_rate', 0.0)
                for opt_id, stats in option_stats.items()
            }
            self.option_success_rates[regime].append(success_rates)

        # Check if this is near a transition
        if self._is_near_boundary(episode, window=10):
            self._update_transition_metrics(episode)

    def _is_near_boundary(self, episode: int, window: int = 10) -> bool:
        """Check if episode is within window of a regime boundary"""
        for boundary in self.regime_boundaries:
            if abs(episode - boundary) <= window:
                return True
        return False

    def _update_transition_metrics(self, episode: int):
        """Update metrics around regime transitions"""
        for boundary in self.regime_boundaries:
            # Pre-transition window: boundary - 20 to boundary - 1
            if episode == boundary - 1:
                pre_window_returns = self.episode_returns[-20:] if len(self.episode_returns) >= 20 else self.episode_returns
                self.pre_transition_performance[boundary] = np.mean(pre_window_returns)

            # Post-transition window: boundary to boundary + 20
            if episode == boundary + 19:
                post_window = self.episode_returns[-20:]
                self.post_transition_performance[boundary] = np.mean(post_window)

                # Compute adaptation time
                self._compute_adaptation_time(boundary)

    def _compute_adaptation_time(self, boundary: int):
        """
        Compute episodes needed to adapt to new regime.

        Adaptation = first episode where moving average >= 90% of pre-transition perf
        """
        if boundary not in self.pre_transition_performance:
            return

        pre_perf = self.pre_transition_performance[boundary]
        threshold = 0.9 * pre_perf
        window_size = 5

        # Look at post-transition episodes
        start_idx = boundary
        for i in range(start_idx, min(start_idx + 50, len(self.episode_returns))):
            if i - window_size + 1 >= start_idx:
                window = self.episode_returns[i - window_size + 1:i + 1]
                if np.mean(window) >= threshold:
                    self.adaptation_times[boundary] = i - boundary + 1
                    return

        # If never recovered, set to max
        self.adaptation_times[boundary] = 50

    def get_regime_average_return(self, regime: str) -> float:
        """Get average training return for a regime"""
        if regime not in self.regime_returns or len(self.regime_returns[regime]) == 0:
            return 0.0
        return float(np.mean(self.regime_returns[regime]))

    def get_forward_transfer(self, boundary_episode: int, window: int = 5) -> float:
        """
        Measure forward transfer: initial performance on new regime.

        Returns: Average return in first 'window' episodes of new regime
        """
        if boundary_episode >= len(self.episode_returns):
            return 0.0

        post_window = self.episode_returns[
            boundary_episode:min(boundary_episode + window, len(self.episode_returns))
        ]
        return float(np.mean(post_window)) if post_window else 0.0

    def get_backward_transfer(self, first_regime: str, later_episode: int, window: int = 10) -> float:
        """
        Measure backward transfer: forgetting of first regime.

        Compare:
        - Final performance on first_regime
        - Performance at later_episode (after seeing other regimes)

        Positive = no forgetting, Negative = catastrophic forgetting
        """
        # Get final performance on first regime
        first_regime_returns = self.regime_returns[first_regime]
        if len(first_regime_returns) == 0:
            return 0.0

        final_first_perf = np.mean(first_regime_returns[-window:])

        # Get performance at later episode (should be in first regime again)
        if later_episode >= len(self.episode_returns):
            return 0.0

        later_perf = np.mean(
            self.episode_returns[max(0, later_episode - window):later_episode + 1]
        )

        # Positive = improvement/retention, Negative = forgetting
        return float(later_perf - final_first_perf)

    def get_average_adaptation_time(self) -> float:
        """Get average adaptation time across all transitions"""
        if len(self.adaptation_times) == 0:
            return 0.0
        return float(np.mean(list(self.adaptation_times.values())))

    def get_regime_summary(self) -> Dict:
        """Get summary statistics per regime"""
        summary = {}
        for regime_name, _, _ in self.regime_schedule:
            returns = self.regime_returns.get(regime_name, [])
            eval_returns = self.regime_eval_returns.get(regime_name, [])

            summary[regime_name] = {
                'avg_return': float(np.mean(returns)) if returns else 0.0,
                'std_return': float(np.std(returns)) if returns else 0.0,
                'avg_eval_return': float(np.mean(eval_returns)) if eval_returns else 0.0,
                'num_episodes': len(returns),
            }

        return summary

    def get_transition_summary(self) -> Dict:
        """Get summary of regime transitions"""
        summary = {
            'boundaries': self.regime_boundaries,
            'pre_transition_perf': self.pre_transition_performance,
            'post_transition_perf': self.post_transition_performance,
            'adaptation_times': self.adaptation_times,
            'avg_adaptation_time': self.get_average_adaptation_time(),
        }
        return summary

    def get_full_report(self) -> Dict:
        """Get comprehensive continual learning report"""
        return {
            'regime_summary': self.get_regime_summary(),
            'transition_summary': self.get_transition_summary(),
            'total_episodes': len(self.episode_returns),
            'overall_avg_return': float(np.mean(self.episode_returns)) if self.episode_returns else 0.0,
        }

    def save(self, filepath: str):
        """Save metrics to JSON file"""
        data = {
            'regime_schedule': [(r, int(s), int(e)) for r, s, e in self.regime_schedule],
            'episode_returns': [float(x) for x in self.episode_returns],
            'episode_lengths': [int(x) for x in self.episode_lengths],
            'episode_regimes': self.episode_regimes,
            'regime_returns': {k: [float(x) for x in v] for k, v in self.regime_returns.items()},
            'regime_eval_returns': {k: [float(x) for x in v] for k, v in self.regime_eval_returns.items()},
            'pre_transition_performance': {int(k): float(v) for k, v in self.pre_transition_performance.items()},
            'post_transition_performance': {int(k): float(v) for k, v in self.post_transition_performance.items()},
            'adaptation_times': {int(k): int(v) for k, v in self.adaptation_times.items()},
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'ContinualMetrics':
        """Load metrics from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct regime schedule
        regime_schedule = [(r, s, e) for r, s, e in data['regime_schedule']]

        # Create instance
        metrics = cls(regime_schedule)

        # Restore data
        metrics.episode_returns = data['episode_returns']
        metrics.episode_lengths = data['episode_lengths']
        metrics.episode_regimes = data['episode_regimes']
        metrics.regime_returns = defaultdict(list, {k: v for k, v in data['regime_returns'].items()})
        metrics.regime_eval_returns = defaultdict(list, {k: v for k, v in data['regime_eval_returns'].items()})
        metrics.pre_transition_performance = {int(k): v for k, v in data['pre_transition_performance'].items()}
        metrics.post_transition_performance = {int(k): v for k, v in data['post_transition_performance'].items()}
        metrics.adaptation_times = {int(k): v for k, v in data['adaptation_times'].items()}

        return metrics
