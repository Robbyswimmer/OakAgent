"""
ARC (Abstract Reasoning Challenge) Environment for OaK Framework

State representation: Flattened grid + task context embeddings
Action space: Object-level transformations (discrete actions)
Reward: Sparse (1.0 for correct solution, 0.0 otherwise)

ARC tasks are grid transformation puzzles where the agent must:
1. Observe training example pairs (input → output)
2. Apply learned transformation to test input
3. Generate correct test output
"""
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import json
from pathlib import Path

from environments.base_env import OaKEnvironment


class ARCEnvironment(OaKEnvironment):
    """
    ARC environment wrapper for OaK framework.

    State: Flattened current grid + task features
    Actions: Object-level transformations (fill, copy, rotate, etc.)
    """

    # ARC color palette (0-9)
    NUM_COLORS = 10
    MAX_GRID_SIZE = 30  # ARC grids can be up to 30x30

    # Action space: Object-level operations
    # Format: (operation_type, param1, param2, ...)
    ACTION_TYPES = {
        0: 'set_cell',        # Set cell at (x, y) to color
        1: 'fill_region',     # Fill connected region with color
        2: 'copy_pattern',    # Copy pattern from one location to another
        3: 'rotate_pattern',  # Rotate pattern 90/180/270 degrees
        4: 'mirror_h',        # Horizontal mirror
        5: 'mirror_v',        # Vertical mirror
        6: 'submit',          # Submit current grid as solution
    }

    def __init__(self, task_path: Optional[str] = None, max_steps: int = 50):
        """
        Initialize ARC environment.

        Args:
            task_path: Path to ARC task JSON file (optional, for testing)
            max_steps: Maximum steps per episode
        """
        super().__init__()

        self.max_steps = max_steps
        self.task_path = task_path

        # Task data (will be loaded from file or dataset)
        self.training_examples = []  # List of (input_grid, output_grid) pairs
        self.test_input = None       # Test input grid
        self.solution_grid = None    # Test output grid (ground truth)

        # Current episode state
        self.current_grid = None
        self.steps_taken = 0

        # Grid dimensions (set when task is loaded)
        self.grid_height = 0
        self.grid_width = 0

        # Load task if path provided
        if task_path:
            self.load_task(task_path)

        # State dimension: flattened grid + task context
        # Task context: grid_height, grid_width, num_training_examples
        self._state_dim = self.MAX_GRID_SIZE * self.MAX_GRID_SIZE + 3

        # Action dimension: For now, simplified discrete actions
        # Each action modifies a cell or region
        self._action_dim = 100  # Simplified: 10x10 grid positions

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def load_task(self, task_path: str):
        """
        Load ARC task from JSON file.

        ARC task format:
        {
            "train": [{"input": [[...]], "output": [[...]]}],
            "test": [{"input": [[...]], "output": [[...]]}]
        }
        """
        with open(task_path, 'r') as f:
            task_data = json.load(f)

        # Load training examples
        self.training_examples = [
            (np.array(example['input']), np.array(example['output']))
            for example in task_data['train']
        ]

        # Load test case (use first test case)
        test_case = task_data['test'][0]
        self.test_input = np.array(test_case['input'])
        self.solution_grid = np.array(test_case['output'])

        # Set grid dimensions from test input
        self.grid_height, self.grid_width = self.test_input.shape

        print(f"Loaded ARC task: {Path(task_path).name}")
        print(f"  Training examples: {len(self.training_examples)}")
        print(f"  Grid size: {self.grid_height}x{self.grid_width}")

    def reset(self) -> np.ndarray:
        """
        Reset environment for new episode.
        Start with test input grid.
        """
        if self.test_input is None:
            raise RuntimeError("No task loaded. Call load_task() first.")

        # Start with a copy of test input
        self.current_grid = self.test_input.copy()
        self.steps_taken = 0

        # Create state observation
        self.current_state = self._create_state_observation()

        return self.current_state.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action in environment.

        Args:
            action: Discrete action index

        Returns:
            next_state: Next observation
            reward: Scalar reward (1.0 if correct, 0.0 otherwise)
            done: Whether episode has terminated
            info: Additional information
        """
        self.steps_taken += 1

        # Decode action (simplified: action is grid position to flip color)
        # For now, simple action space: flip cell color at position
        row = action // 10
        col = action % 10

        # Ensure action is within grid bounds
        if row < self.grid_height and col < self.grid_width:
            # Flip cell color (cycle through colors)
            self.current_grid[row, col] = (self.current_grid[row, col] + 1) % self.NUM_COLORS

        # Check if solution is correct
        done = False
        reward = 0.0

        if self._is_solution_correct():
            reward = 1.0
            done = True
        elif self.steps_taken >= self.max_steps:
            done = True

        # Update state
        self.current_state = self._create_state_observation()

        info = {
            'steps_taken': self.steps_taken,
            'grid_match_ratio': self._compute_match_ratio(),
        }

        return self.current_state.copy(), reward, done, info

    def _create_state_observation(self) -> np.ndarray:
        """
        Create state observation from current grid and task context.

        Returns:
            state: Flattened state vector
        """
        # Flatten current grid to MAX_GRID_SIZE x MAX_GRID_SIZE (pad with zeros)
        flat_grid = np.zeros((self.MAX_GRID_SIZE, self.MAX_GRID_SIZE))
        flat_grid[:self.grid_height, :self.grid_width] = self.current_grid
        flat_grid = flat_grid.flatten()

        # Task context features
        context = np.array([
            self.grid_height / self.MAX_GRID_SIZE,
            self.grid_width / self.MAX_GRID_SIZE,
            len(self.training_examples) / 10.0,  # Normalize
        ])

        # Concatenate
        state = np.concatenate([flat_grid, context])
        return state.astype(np.float32)

    def _is_solution_correct(self) -> bool:
        """Check if current grid matches solution grid."""
        if self.solution_grid is None:
            return False
        return np.array_equal(self.current_grid, self.solution_grid)

    def _compute_match_ratio(self) -> float:
        """Compute ratio of cells that match solution."""
        if self.solution_grid is None:
            return 0.0

        if self.current_grid.shape != self.solution_grid.shape:
            # Compare overlap for mismatched dimensions
            min_h = min(self.current_grid.shape[0], self.solution_grid.shape[0])
            min_w = min(self.current_grid.shape[1], self.solution_grid.shape[1])
            matches = np.sum(
                self.current_grid[:min_h, :min_w] == self.solution_grid[:min_h, :min_w]
            )
            total = max(self.current_grid.size, self.solution_grid.size)
        else:
            matches = np.sum(self.current_grid == self.solution_grid)
            total = self.solution_grid.size
        return matches / total

    def get_state_components(self, state: Optional[np.ndarray] = None) -> Tuple:
        """
        Extract meaningful components from state for GVF cumulants.

        Returns:
            (grid_entropy, object_count, symmetry_score, match_ratio)
        """
        if state is None:
            grid = self.current_grid
        else:
            # Reconstruct grid from flattened state
            flat_grid = state[:self.MAX_GRID_SIZE * self.MAX_GRID_SIZE]
            grid = flat_grid.reshape(self.MAX_GRID_SIZE, self.MAX_GRID_SIZE)
            grid = grid[:self.grid_height, :self.grid_width]

        # Compute grid entropy
        unique, counts = np.unique(grid, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-8))

        # Count distinct objects (connected components)
        # Simplified: count non-zero cells
        object_count = np.sum(grid > 0)

        # Compute symmetry score (horizontal + vertical)
        h_symmetry = np.mean(grid == np.fliplr(grid))
        v_symmetry = np.mean(grid == np.flipud(grid))
        symmetry_score = (h_symmetry + v_symmetry) / 2.0

        # Compute match ratio with solution
        match_ratio = self._compute_match_ratio()

        return entropy, object_count, symmetry_score, match_ratio

    def close(self):
        """Clean up environment resources."""
        pass

    def render(self, mode: str = 'human'):
        """
        Render current grid state.

        Args:
            mode: Render mode ('human' for console, 'rgb_array' for image)
        """
        if mode == 'human':
            print("\nCurrent Grid:")
            print(self.current_grid)
            print(f"\nMatch ratio: {self._compute_match_ratio():.2%}")

    def get_observation_space_info(self) -> Dict[str, Any]:
        """Get metadata about observation space."""
        return {
            'type': 'grid',
            'shape': (self.MAX_GRID_SIZE, self.MAX_GRID_SIZE),
            'grid_size': (self.grid_height, self.grid_width),
            'num_colors': self.NUM_COLORS,
        }
