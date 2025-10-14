"""
ARC-Specific Option Templates
Temporal abstractions for ARC domain (grid transformations)
"""
import numpy as np
from options.option import Option


class FillRegionOption(Option):
    """
    Fill Region Option: Fill a connected region with a target color
    """

    def __init__(self, option_id, name, state_dim, action_dim,
                 target_color=1, uniformity_threshold=0.9, max_length=15):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.target_color = target_color
        self.uniformity_threshold = uniformity_threshold

    def termination(self, state):
        """Terminate when region is sufficiently uniform"""
        # Extract grid from state (simplified)
        grid_size = 30 * 30
        if len(state) >= grid_size:
            grid = state[:grid_size].reshape(30, 30)

            # Check uniformity: what fraction of cells match target color
            matching_cells = np.sum(grid == self.target_color)
            total_cells = grid.size
            uniformity = matching_cells / total_cells

            achieved = uniformity >= self.uniformity_threshold
            return achieved, achieved

        return False, False

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: shaped to encourage filling region"""
        grid_size = 30 * 30
        if len(state) >= grid_size:
            grid = state[:grid_size].reshape(30, 30)

            # Reward based on fraction matching target color
            matching_cells = np.sum(grid == self.target_color)
            total_cells = grid.size
            reward = (matching_cells / total_cells) - 0.5  # Center around 0

            return reward * 10.0  # Scale for stronger signal

        return 0.0


class SymmetrizeOption(Option):
    """
    Symmetrize Option: Make grid symmetric (horizontal or vertical)
    """

    def __init__(self, option_id, name, state_dim, action_dim,
                 axis='horizontal', symmetry_threshold=0.8, max_length=20):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.axis = axis  # 'horizontal' or 'vertical'
        self.symmetry_threshold = symmetry_threshold

    def termination(self, state):
        """Terminate when grid is sufficiently symmetric"""
        grid_size = 30 * 30
        if len(state) >= grid_size:
            grid = state[:grid_size].reshape(30, 30)

            # Compute symmetry score
            if self.axis == 'horizontal':
                symmetry = np.mean(grid == np.fliplr(grid))
            else:  # vertical
                symmetry = np.mean(grid == np.flipud(grid))

            achieved = symmetry >= self.symmetry_threshold
            return achieved, achieved

        return False, False

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: shaped to encourage symmetry"""
        grid_size = 30 * 30
        if len(state) >= grid_size:
            grid = state[:grid_size].reshape(30, 30)

            # Reward based on symmetry score
            if self.axis == 'horizontal':
                symmetry = np.mean(grid == np.fliplr(grid))
            else:  # vertical
                symmetry = np.mean(grid == np.flipud(grid))

            # Exponential shaping
            distance = 1.0 - symmetry
            reward = np.exp(-5.0 * distance) - 1.0

            # Bonus for achieving goal
            if symmetry >= self.symmetry_threshold:
                reward += 2.0

            return reward * 5.0

        return 0.0


class ReduceEntropyOption(Option):
    """
    Reduce Entropy Option: Simplify grid by reducing color diversity
    """

    def __init__(self, option_id, name, state_dim, action_dim,
                 max_entropy=1.5, max_length=15):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.max_entropy = max_entropy

    def termination(self, state):
        """Terminate when grid entropy is below threshold"""
        grid_size = 30 * 30
        if len(state) >= grid_size:
            grid = state[:grid_size].reshape(30, 30)

            # Compute entropy
            unique, counts = np.unique(grid, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-8))

            achieved = entropy <= self.max_entropy
            return achieved, achieved

        return False, False

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: shaped to encourage entropy reduction"""
        grid_size = 30 * 30
        if len(state) >= grid_size:
            grid = state[:grid_size].reshape(30, 30)

            # Compute entropy
            unique, counts = np.unique(grid, return_counts=True)
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-8))

            # Reward for lower entropy
            reward = (2.5 - entropy) / 2.5  # Normalize to [-1, 1] range

            return reward * 5.0

        return 0.0


class CopyPatternOption(Option):
    """
    Copy Pattern Option: Copy a pattern from one region to another
    (Simplified version - placeholder for more complex implementation)
    """

    def __init__(self, option_id, name, state_dim, action_dim,
                 max_length=20):
        super().__init__(option_id, name, state_dim, action_dim, max_length)

    def termination(self, state):
        """Terminate after executing pattern copy (placeholder)"""
        # For now, terminate based on step count
        # In full implementation, would check if pattern was successfully copied
        return False, False

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: placeholder for pattern matching reward"""
        return 0.0


class MatchSolutionOption(Option):
    """
    Match Solution Option: Move grid closer to solution
    (High-level option that uses similarity metric)
    """

    def __init__(self, option_id, name, state_dim, action_dim,
                 similarity_threshold=0.95, max_length=25):
        super().__init__(option_id, name, state_dim, action_dim, max_length)
        self.similarity_threshold = similarity_threshold

    def termination(self, state):
        """Terminate when grid is very similar to solution"""
        # Extract similarity from state components (if encoded)
        # Placeholder: would need actual similarity computation
        return False, False

    def compute_pseudo_reward(self, state):
        """Pseudo-reward: shaped to encourage matching solution"""
        # Placeholder: would compute actual similarity
        # For now, return neutral reward
        return 0.0
