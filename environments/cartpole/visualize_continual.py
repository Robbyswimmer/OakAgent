"""
Continual Learning Visualization Tools
Plot learning curves, adaptation metrics, and comparative analysis
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os


# Regime colors for consistent visualization
REGIME_COLORS = {
    'R1_base': '#3498db',      # Blue
    'R2_heavy': '#e74c3c',     # Red
    'R3_long': '#2ecc71',      # Green
    'R4_friction': '#f39c12',  # Orange
    'R5_gravity': '#9b59b6',   # Purple
}


def load_seed_results(results_dir: str) -> Tuple[List[Dict], Dict, Dict]:
    """
    Load all seed results from a continual sweep directory.

    Returns:
        seed_metrics: List of metrics dicts (one per seed)
        aggregated: Aggregated results dict
        config: Config dict
    """
    results_path = Path(results_dir)

    # Load config
    config_path = results_path / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load aggregated results
    agg_path = results_path / "aggregated_results.json"
    with open(agg_path, 'r') as f:
        aggregated = json.load(f)

    # Load per-seed results
    seed_metrics = []
    seed_dirs = sorted([d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("seed_")])

    for seed_dir in seed_dirs:
        metrics_path = seed_dir / "continual_metrics.json"
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # Also load episode returns
        returns_path = seed_dir / "episode_returns.npy"
        if returns_path.exists():
            metrics['episode_returns_array'] = np.load(returns_path)

        seed_metrics.append(metrics)

    return seed_metrics, aggregated, config


def plot_learning_curves(seed_metrics: List[Dict], config: Dict,
                         output_path: str, show_individual_seeds: bool = True):
    """
    Plot learning curves with regime boundaries.

    Args:
        seed_metrics: List of metrics dicts from each seed
        config: Config dict
        output_path: Path to save figure
        show_individual_seeds: If True, plot individual seed curves (lighter lines)
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Extract regime schedule
    regime_schedule = config['REGIME_SCHEDULE']
    regime_names = [r[0] for r in regime_schedule]
    regime_boundaries = [r[1] for r in regime_schedule if r[1] > 0]

    num_episodes = len(seed_metrics[0]['episode_returns'])

    # Plot individual seeds (light lines)
    if show_individual_seeds:
        for i, metrics in enumerate(seed_metrics):
            returns = metrics['episode_returns']
            ax.plot(returns, alpha=0.2, linewidth=0.8, color='gray')

    # Compute mean and std across seeds
    all_returns = np.array([m['episode_returns'] for m in seed_metrics])
    mean_returns = np.mean(all_returns, axis=0)
    std_returns = np.std(all_returns, axis=0)

    # Plot mean with confidence band
    episodes = np.arange(num_episodes)
    ax.plot(episodes, mean_returns, linewidth=2.5, color='black', label='Mean across seeds')
    ax.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns,
                     alpha=0.3, color='gray', label='±1 std')

    # Add regime boundaries and background colors
    for i, (regime_name, start_ep, end_ep) in enumerate(regime_schedule):
        # Background color for regime
        color = REGIME_COLORS.get(regime_name, '#ecf0f1')
        ax.axvspan(start_ep, end_ep, alpha=0.1, color=color)

        # Vertical line at transition
        if start_ep > 0:
            ax.axvline(start_ep, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Regime label
        mid_ep = (start_ep + end_ep) / 2
        ax.text(mid_ep, ax.get_ylim()[1] * 0.95, regime_name,
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))

    # Styling
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title('Continual Learning: Episode Returns Across Regimes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved learning curves to {output_path}")
    plt.close()


def plot_adaptation_times(seed_metrics: List[Dict], config: Dict, output_path: str):
    """
    Plot adaptation time for each regime transition.

    Args:
        seed_metrics: List of metrics dicts
        config: Config dict
        output_path: Path to save figure
    """
    regime_schedule = config['REGIME_SCHEDULE']
    regime_boundaries = [r[1] for r in regime_schedule if r[1] > 0]

    if len(regime_boundaries) == 0:
        print("No regime transitions to plot adaptation times")
        return

    # Extract adaptation times for each boundary across seeds
    adaptation_data = {boundary: [] for boundary in regime_boundaries}

    for metrics in seed_metrics:
        adaptation_times = metrics.get('adaptation_times', {})
        for boundary_str, time in adaptation_times.items():
            boundary = int(boundary_str)
            if boundary in adaptation_data:
                adaptation_data[boundary].append(time)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    boundaries_list = sorted(adaptation_data.keys())
    means = [np.mean(adaptation_data[b]) if adaptation_data[b] else 0 for b in boundaries_list]
    stds = [np.std(adaptation_data[b]) if adaptation_data[b] else 0 for b in boundaries_list]

    # Regime labels for transitions
    transition_labels = []
    for i, boundary in enumerate(boundaries_list):
        # Find which regime this boundary starts
        for j, (regime_name, start_ep, end_ep) in enumerate(regime_schedule):
            if start_ep == boundary:
                if j > 0:
                    prev_regime = regime_schedule[j-1][0]
                    transition_labels.append(f"{prev_regime}\n→\n{regime_name}")
                else:
                    transition_labels.append(f"Start\n→\n{regime_name}")
                break

    x_pos = np.arange(len(boundaries_list))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue', edgecolor='black')

    # Color bars by target regime
    for i, boundary in enumerate(boundaries_list):
        for regime_name, start_ep, end_ep in regime_schedule:
            if start_ep == boundary:
                bars[i].set_color(REGIME_COLORS.get(regime_name, 'steelblue'))
                break

    ax.set_xticks(x_pos)
    ax.set_xticklabels(transition_labels, fontsize=9)
    ax.set_ylabel('Adaptation Time (episodes)', fontsize=12)
    ax.set_xlabel('Regime Transition', fontsize=12)
    ax.set_title('Adaptation Time at Each Regime Transition', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add horizontal line at 50 (max adaptation time)
    ax.axhline(50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Max (did not adapt)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved adaptation times to {output_path}")
    plt.close()


def plot_regime_performance_comparison(seed_metrics: List[Dict], config: Dict, output_path: str):
    """
    Compare average performance across regimes.

    Args:
        seed_metrics: List of metrics dicts
        config: Config dict
        output_path: Path to save figure
    """
    regime_schedule = config['REGIME_SCHEDULE']
    regime_names = [r[0] for r in regime_schedule]

    # Collect per-regime returns
    regime_data = {regime: [] for regime in regime_names}

    for metrics in seed_metrics:
        regime_returns = metrics.get('regime_returns', {})
        for regime in regime_names:
            if regime in regime_returns:
                regime_data[regime].append(np.mean(regime_returns[regime]))

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    means = [np.mean(regime_data[r]) if regime_data[r] else 0 for r in regime_names]
    stds = [np.std(regime_data[r]) if regime_data[r] else 0 for r in regime_names]
    colors = [REGIME_COLORS.get(r, 'steelblue') for r in regime_names]

    x_pos = np.arange(len(regime_names))
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=colors, edgecolor='black')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(regime_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Average Episode Return', fontsize=12)
    ax.set_xlabel('Regime', fontsize=12)
    ax.set_title('Performance Comparison Across Regimes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved regime comparison to {output_path}")
    plt.close()


def plot_windowed_performance(seed_metrics: List[Dict], config: Dict,
                              output_path: str, window: int = 10):
    """
    Plot rolling average of returns with regime boundaries.

    Args:
        seed_metrics: List of metrics dicts
        config: Config dict
        output_path: Path to save figure
        window: Window size for moving average
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    regime_schedule = config['REGIME_SCHEDULE']
    regime_boundaries = [r[1] for r in regime_schedule if r[1] > 0]

    # Compute rolling average across seeds
    all_returns = np.array([m['episode_returns'] for m in seed_metrics])

    # Apply rolling window
    def rolling_mean(data, window):
        cumsum = np.cumsum(data, axis=1)
        cumsum[:, window:] = cumsum[:, window:] - cumsum[:, :-window]
        return cumsum[:, window - 1:] / window

    smoothed_returns = rolling_mean(all_returns, window)
    mean_smoothed = np.mean(smoothed_returns, axis=0)
    std_smoothed = np.std(smoothed_returns, axis=0)

    episodes = np.arange(window - 1, len(seed_metrics[0]['episode_returns']))

    # Plot
    ax.plot(episodes, mean_smoothed, linewidth=2.5, color='darkblue', label=f'{window}-episode moving avg')
    ax.fill_between(episodes, mean_smoothed - std_smoothed, mean_smoothed + std_smoothed,
                     alpha=0.3, color='lightblue', label='±1 std')

    # Add regime boundaries
    for regime_name, start_ep, end_ep in regime_schedule:
        color = REGIME_COLORS.get(regime_name, '#ecf0f1')
        ax.axvspan(start_ep, end_ep, alpha=0.1, color=color)

        if start_ep > 0:
            ax.axvline(start_ep, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        mid_ep = (start_ep + end_ep) / 2
        ax.text(mid_ep, ax.get_ylim()[1] * 0.95, regime_name,
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.5))

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(f'{window}-Episode Moving Average Return', fontsize=12)
    ax.set_title('Smoothed Performance Across Regimes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved windowed performance to {output_path}")
    plt.close()


def create_summary_report(seed_metrics: List[Dict], aggregated: Dict,
                         config: Dict, output_path: str):
    """
    Create a text summary report of the continual learning results.

    Args:
        seed_metrics: List of metrics dicts
        aggregated: Aggregated results dict
        config: Config dict
        output_path: Path to save report
    """
    regime_schedule = config['REGIME_SCHEDULE']
    regime_names = [r[0] for r in regime_schedule]

    report = []
    report.append("="*70)
    report.append("CONTINUAL LEARNING EXPERIMENT SUMMARY")
    report.append("="*70)
    report.append("")

    # Configuration
    report.append("Configuration:")
    report.append(f"  Total episodes: {config['NUM_EPISODES']}")
    report.append(f"  Number of seeds: {aggregated['num_seeds']}")
    report.append(f"  Regimes: {' → '.join(regime_names)}")
    report.append("")

    # Overall performance
    report.append("Overall Performance:")
    report.append(f"  Mean return: {aggregated['overall_avg_return_mean']:.2f} ± {aggregated['overall_avg_return_std']:.2f}")
    report.append("")

    # Per-regime performance
    report.append("Per-Regime Performance:")
    report.append("-"*70)
    for regime in regime_names:
        if regime in aggregated['regime_performance']:
            stats = aggregated['regime_performance'][regime]
            report.append(f"  {regime}:")
            report.append(f"    Training return: {stats['avg_return_mean']:.2f} ± {stats['avg_return_std']:.2f}")
            report.append(f"    Eval return:     {stats['avg_eval_return_mean']:.2f} ± {stats['avg_eval_return_std']:.2f}")
    report.append("")

    # Adaptation metrics
    if aggregated['adaptation_time_mean'] is not None:
        report.append("Adaptation Metrics:")
        report.append(f"  Mean adaptation time: {aggregated['adaptation_time_mean']:.2f} ± {aggregated['adaptation_time_std']:.2f} episodes")
        report.append("  (Time to recover 90% of pre-transition performance)")
        report.append("")

    # Regime-specific notes
    report.append("Regime Characteristics:")
    regime_descriptions = {
        'R1_base': 'Baseline CartPole (standard parameters)',
        'R2_heavy': 'Heavy pole (5× mass)',
        'R3_long': 'Long pole (2× length)',
        'R4_friction': 'Added friction (0.95 damping)',
        'R5_gravity': 'High gravity (1.22× Earth)',
    }
    for regime in regime_names:
        if regime in regime_descriptions:
            report.append(f"  {regime}: {regime_descriptions[regime]}")
    report.append("")

    report.append("="*70)

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"Saved summary report to {output_path}")


def visualize_all(results_dir: str, output_dir: Optional[str] = None):
    """
    Generate all visualizations for a continual learning experiment.

    Args:
        results_dir: Directory containing continual sweep results
        output_dir: Directory to save plots (default: results_dir/plots)
    """
    print(f"\n{'='*70}")
    print("GENERATING CONTINUAL LEARNING VISUALIZATIONS")
    print(f"{'='*70}\n")

    # Load data
    print(f"Loading results from {results_dir}...")
    seed_metrics, aggregated, config = load_seed_results(results_dir)
    print(f"Loaded {len(seed_metrics)} seeds")

    # Output directory
    if output_dir is None:
        output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    plot_learning_curves(
        seed_metrics, config,
        os.path.join(output_dir, "learning_curves.png"),
        show_individual_seeds=True
    )

    plot_windowed_performance(
        seed_metrics, config,
        os.path.join(output_dir, "smoothed_performance.png"),
        window=10
    )

    plot_adaptation_times(
        seed_metrics, config,
        os.path.join(output_dir, "adaptation_times.png")
    )

    plot_regime_performance_comparison(
        seed_metrics, config,
        os.path.join(output_dir, "regime_comparison.png")
    )

    # Create summary report
    create_summary_report(
        seed_metrics, aggregated, config,
        os.path.join(output_dir, "summary_report.txt")
    )

    print(f"\n{'='*70}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize continual learning results')

    parser.add_argument('results_dir', type=str,
                       help='Directory containing continual sweep results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: results_dir/plots)')

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    visualize_all(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
