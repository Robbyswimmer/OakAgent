"""
Visualization utilities for OaK-CartPole results
Plots learning curves, model errors, and option statistics
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_learning_curve(episode_returns, save_path='results/learning_curve.png'):
    """Plot episode returns over time"""
    plt.figure(figsize=(10, 6))

    # Raw returns
    plt.plot(episode_returns, alpha=0.3, label='Episode Return')

    # Smoothed curve (moving average)
    window = 10
    if len(episode_returns) >= window:
        smoothed = np.convolve(episode_returns, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_returns)), smoothed,
                linewidth=2, label=f'Moving Avg ({window})')

    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('OaK-CartPole Learning Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved learning curve to {save_path}")


def plot_fc_stomp_activity(fc_stomp_history, save_path='results/fc_stomp_activity.png'):
    """Plot FC-STOMP activity over time"""
    if not fc_stomp_history:
        print("No FC-STOMP history to plot")
        return

    steps = [h['step'] for h in fc_stomp_history]
    features = [h['features_mined'] for h in fc_stomp_history]
    subtasks = [h['subtasks_formed'] for h in fc_stomp_history]
    options_created = [h['options_created'] for h in fc_stomp_history]
    options_pruned = [h['options_pruned'] for h in fc_stomp_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(steps, features, marker='o')
    axes[0, 0].set_title('Features Mined')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, subtasks, marker='s', color='orange')
    axes[0, 1].set_title('Subtasks Formed')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, options_created, marker='^', color='green')
    axes[1, 0].set_title('Options Created')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, options_pruned, marker='v', color='red')
    axes[1, 1].set_title('Options Pruned')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('FC-STOMP Activity Over Time')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved FC-STOMP activity to {save_path}")


def plot_option_statistics(option_stats, save_path='results/option_stats.png'):
    """Plot option execution statistics"""
    if not option_stats:
        print("No option statistics to plot")
        return

    option_ids = list(option_stats.keys())
    option_names = [f"Option {oid}" for oid in option_ids]

    executions = [option_stats[oid]['executions'] for oid in option_ids]
    success_rates = [option_stats[oid]['success_rate'] * 100 for oid in option_ids]
    avg_durations = [option_stats[oid]['avg_duration_steps'] for oid in option_ids]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Executions
    axes[0].bar(option_names, executions, color='skyblue')
    axes[0].set_title('Option Execution Count')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Success rates
    axes[1].bar(option_names, success_rates, color='lightgreen')
    axes[1].set_title('Option Success Rate')
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    axes[1].legend()

    # Average duration
    axes[2].bar(option_names, avg_durations, color='coral')
    axes[2].set_title('Average Option Duration (steps)')
    axes[2].set_ylabel('Steps')
    axes[2].tick_params(axis='x', rotation=45)

    plt.suptitle('Option Statistics')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved option statistics to {save_path}")


def visualize_results(results_path='results/oak_cartpole_results.json'):
    """Main visualization function"""
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, 'r') as f:
        results = json.load(f)

    os.makedirs('results', exist_ok=True)

    # Plot learning curve
    plot_learning_curve(results['episode_returns'])

    # Plot FC-STOMP activity
    if 'fc_stomp_history' in results:
        plot_fc_stomp_activity(results['fc_stomp_history'])

    # Plot option statistics
    if 'option_stats' in results:
        plot_option_statistics(results['option_stats'])

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Total Episodes: {len(results['episode_returns'])}")
    print(f"Final Evaluation Return: {results.get('final_eval_return', 'N/A'):.1f}")
    print(f"Best Episode Return: {max(results['episode_returns']):.1f}")
    print(f"Average Return (last 100): {np.mean(results['episode_returns'][-100:]):.1f}")

    if 'planner_stats' in results:
        print(f"\nPlanner Statistics:")
        for key, value in results['planner_stats'].items():
            print(f"  {key}: {value}")

    print("\nAll visualizations saved to results/")


if __name__ == "__main__":
    visualize_results()
