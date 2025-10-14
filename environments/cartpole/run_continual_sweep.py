"""
Multi-Seed Continual Learning Runner
Runs continual learning experiments across multiple seeds for statistical significance
"""
import argparse
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
import random

from continual_config import ContinualConfig, TwoRegimeConfig, FastAdaptationConfig
from main import OaKAgent


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # Note: PyTorch seeds set in agent if needed


def run_single_seed(config, seed: int, output_dir: str):
    """
    Run single continual learning experiment with given seed.

    Args:
        config: Config object with REGIME_SCHEDULE
        seed: Random seed
        output_dir: Directory to save results

    Returns:
        Dict with metrics
    """
    print(f"\n{'='*60}")
    print(f"Running seed {seed}")
    print(f"{'='*60}\n")

    set_seed(seed)

    # Create agent with continual environment
    agent = OaKAgent(config, use_continual_env=True, initial_regime='R1_base')

    # Train
    agent.train_continual(num_episodes=config.NUM_EPISODES)

    # Get metrics
    metrics = agent.continual_metrics.get_full_report()

    # Save
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(seed_dir, "continual_metrics.json")
    agent.continual_metrics.save(metrics_path)

    # Save episode returns for plotting
    returns_path = os.path.join(seed_dir, "episode_returns.npy")
    np.save(returns_path, np.array(agent.episode_returns))

    print(f"\nSeed {seed} complete. Results saved to {seed_dir}")

    return metrics


def aggregate_results(all_metrics, output_dir: str):
    """
    Aggregate results across seeds.

    Args:
        all_metrics: List of metrics dicts from each seed
        output_dir: Directory to save aggregated results
    """
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}\n")

    num_seeds = len(all_metrics)

    # Aggregate regime-level performance
    regime_aggregates = {}

    # Get all regime names
    regime_names = list(all_metrics[0]['regime_summary'].keys())

    for regime in regime_names:
        avg_returns = [m['regime_summary'][regime]['avg_return'] for m in all_metrics]
        eval_returns = [m['regime_summary'][regime]['avg_eval_return'] for m in all_metrics]

        regime_aggregates[regime] = {
            'avg_return_mean': float(np.mean(avg_returns)),
            'avg_return_std': float(np.std(avg_returns)),
            'avg_eval_return_mean': float(np.mean(eval_returns)),
            'avg_eval_return_std': float(np.std(eval_returns)),
        }

    # Aggregate adaptation times
    adaptation_times = []
    for m in all_metrics:
        if 'transition_summary' in m and 'avg_adaptation_time' in m['transition_summary']:
            adaptation_times.append(m['transition_summary']['avg_adaptation_time'])

    aggregated = {
        'num_seeds': num_seeds,
        'regime_performance': regime_aggregates,
        'adaptation_time_mean': float(np.mean(adaptation_times)) if adaptation_times else None,
        'adaptation_time_std': float(np.std(adaptation_times)) if adaptation_times else None,
        'overall_avg_return_mean': float(np.mean([m['overall_avg_return'] for m in all_metrics])),
        'overall_avg_return_std': float(np.std([m['overall_avg_return'] for m in all_metrics])),
    }

    # Save
    agg_path = os.path.join(output_dir, "aggregated_results.json")
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (mean ± std across seeds)")
    print("="*60)

    for regime in regime_names:
        stats = regime_aggregates[regime]
        print(f"\n{regime}:")
        print(f"  Avg Return: {stats['avg_return_mean']:.1f} ± {stats['avg_return_std']:.1f}")
        print(f"  Eval Return: {stats['avg_eval_return_mean']:.1f} ± {stats['avg_eval_return_std']:.1f}")

    if adaptation_times:
        print(f"\nAverage Adaptation Time: {aggregated['adaptation_time_mean']:.1f} ± {aggregated['adaptation_time_std']:.1f} episodes")

    print(f"\nOverall Avg Return: {aggregated['overall_avg_return_mean']:.1f} ± {aggregated['overall_avg_return_std']:.1f}")
    print("="*60 + "\n")

    return aggregated


def run_continual_sweep(config, num_seeds: int, output_dir: str):
    """
    Run continual learning experiment across multiple seeds.

    Args:
        config: Config object
        num_seeds: Number of random seeds to run
        output_dir: Directory to save all results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.get_config_dict(), f, indent=2)

    # Run seeds
    all_metrics = []
    for seed in range(num_seeds):
        metrics = run_single_seed(config, seed, output_dir)
        all_metrics.append(metrics)

    # Aggregate
    aggregated = aggregate_results(all_metrics, output_dir)

    print(f"\nAll results saved to: {output_dir}")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Run continual learning sweep')

    parser.add_argument('--config', type=str, default='continual',
                        choices=['continual', 'two_regime', 'fast'],
                        help='Config to use')
    parser.add_argument('--num_seeds', type=int, default=5,
                        help='Number of seeds to run')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')

    args = parser.parse_args()

    # Select config
    if args.config == 'continual':
        config = ContinualConfig()
    elif args.config == 'two_regime':
        config = TwoRegimeConfig()
    elif args.config == 'fast':
        config = FastAdaptationConfig()
    else:
        raise ValueError(f"Unknown config: {args.config}")

    # Output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"continual_results/{args.config}_{timestamp}"

    print(f"\n{'='*60}")
    print("CONTINUAL LEARNING SWEEP")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Num seeds: {args.num_seeds}")
    print(f"Output dir: {args.output_dir}")
    print(f"Regimes: {[r for r, _, _ in config.REGIME_SCHEDULE]}")
    print(f"{'='*60}\n")

    # Run
    run_continual_sweep(config, args.num_seeds, args.output_dir)


if __name__ == "__main__":
    main()
