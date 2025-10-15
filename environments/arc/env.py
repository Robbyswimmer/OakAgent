# -*- coding: utf-8 -*-
"""
ARC (Abstract Reasoning Challenge) Environment for OaK Framework

State representation: Flattened grid + exemplar embeddings + spatial features
Action space: Structured object-level transformations (paint, fill, copy, rotate, submit)
Reward: Dense shaping on match ratio with bonus for correct submissions

ARC tasks are grid transformation puzzles where the agent must:
1. Observe training example pairs (input → output)
2. Apply learned transformation to test input
3. Generate correct test output
"""
import math
import random
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import json
from pathlib import Path

from environments.base_env import OaKEnvironment


class ARCEnvironment(OaKEnvironment):
    """
    ARC environment wrapper for OaK framework.

    State: Flattened current grid + exemplar encodings + spatial features
    Actions: Object-level transformations (fill, copy, rotate, exemplar copy, etc.)
    """

    # ARC color palette (0-9)
    NUM_COLORS = 10
    MAX_GRID_SIZE = 30  # ARC grids can be up to 30x30
    DEFAULT_MAX_TRAINING_EXAMPLES = 7  # ARC tasks often include up to 7 exemplars
    ACTION_COMPONENT_DIM = 6  # operator + (grid_from, grid_to, color, exemplar, orientation)
    HIERARCHICAL_OPERATORS = [
        'paint_cell',
        'fill_region',
        'copy_patch',
        'copy_exemplar_patch',
        'stamp_exemplar_output',
        'mirror',
        'rotate',
        'submit',
    ]

    # Parameters that control how many structured actions we expose.
    # Keeping them relatively small still gives us far richer behaviour
    # than the original "flip a single cell" space while remaining tractable
    # for small-scale experiments.
    COPY_PATCH_SIZE = 3
    COPY_PATCH_STRIDE = 3

    def __init__(self,
                 task_path: Optional[str] = None,
                 max_steps: int = 50,
                 max_training_examples: Optional[int] = None,
                 reward_mode: str = "binary",
                 working_grid_size: int = 16,
                 action_stride: int = 3,
                 max_paint_actions: int = 1000,
                 max_fill_actions: int = 500,
                 max_copy_actions: int = 500,
                 max_exemplar_actions: int = 1000,
                 mode: str = "train") -> None:
        """
        Initialize ARC environment.

        Args:
            task_path: Path to ARC task JSON file (optional, for testing)
            max_steps: Maximum steps per episode
        """
        super().__init__()

        self.max_steps = max_steps
        self.task_path = task_path

        self.working_grid_size = int(working_grid_size)
        self.action_stride = max(1, int(action_stride))
        self.max_paint_actions = max(0, int(max_paint_actions))
        self.max_fill_actions = max(0, int(max_fill_actions))
        self.max_copy_actions = max(0, int(max_copy_actions))
        self.max_exemplar_actions = max(0, int(max_exemplar_actions))
        self.mode = mode

        if reward_mode not in {"binary", "dense"}:
            raise ValueError("reward_mode must be 'binary' or 'dense'")
        self.reward_mode = reward_mode

        # Task data (will be loaded from file or dataset)
        self.training_examples = []  # List of (input_grid, output_grid) pairs
        self.test_input = None       # Test input grid
        self.solution_grid = None    # Test output grid (ground truth)

        # Current episode state
        self.current_grid = None
        self.steps_taken = 0
        self.prev_match_ratio = 0.0

        # Grid dimensions (set when task is loaded)
        self.grid_height = 0
        self.grid_width = 0

        # Cached exemplar encodings (rebuilt when a new task is loaded)
        self.max_training_examples = (
            max_training_examples
            if max_training_examples is not None
            else self.DEFAULT_MAX_TRAINING_EXAMPLES
        )

        # Derived feature sizes
        self._grid_flat_dim = self.working_grid_size * self.working_grid_size
        self._spatial_feature_dim = len(
            self._extract_spatial_features(np.zeros((1, 1), dtype=np.float32))
        )
        # Context terms: height, width, exemplar count, steps remaining,
        # match ratio, entropy, object fraction, symmetry
        self._context_dim = 8

        self._action_table: List[Dict[str, Any]] = []
        self._action_dim = 0
        self.operator_names = list(self.HIERARCHICAL_OPERATORS)
        self.action_component_dim = self.ACTION_COMPONENT_DIM
        self._structured_lookup: Dict[Tuple[Any, ...], int] = {}

        self._set_feature_buffers()

        if task_path:
            self.load_task(task_path)

    def set_mode(self, mode: str):
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be 'train' or 'eval'")
        self.mode = mode

    def _set_feature_buffers(self):
        """Allocate buffers whose sizes depend on exemplar capacity."""

        self._task_feature_dim = (
            self.max_training_examples * (2 * self.NUM_COLORS + 3)
        )
        self._cached_task_features = np.zeros(self._task_feature_dim, dtype=np.float32)
        grid_flat_dim = self.working_grid_size * self.working_grid_size
        self._grid_flat_dim = grid_flat_dim
        self._state_dim = (
            grid_flat_dim
            + self._spatial_feature_dim
            + self._task_feature_dim
            + self._context_dim
        )

    def _ensure_exemplar_capacity(self, count: int):
        """Grow exemplar buffers if a task provides more examples than expected."""

        if count <= self.max_training_examples:
            return

        self.max_training_examples = count
        self._set_feature_buffers()

        # Action table populated once the task dimensions are known
        self._action_table = []
        self._action_dim = 0

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        if not self._action_table:
            self._rebuild_action_table()
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

        self._ensure_exemplar_capacity(len(self.training_examples))

        # Load test case (use first test case)
        test_case = task_data['test'][0]
        self.test_input = np.array(test_case['input'])
        self.solution_grid = np.array(test_case['output'])

        # Set grid dimensions from test input
        self.grid_height, self.grid_width = self.test_input.shape

        # Pre-compute exemplar embedding for richer task conditioning
        self._cached_task_features = self._compute_task_context_features()
        self._cached_task_features = self._compute_task_context_features()

        # Rebuild the structured action table now that we know task dimensions
        self._rebuild_action_table()

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

        # Rebuild action table in case caller changed task data manually
        self._rebuild_action_table()

        # Start with a copy of test input
        self.current_grid = self.test_input.copy()
        self.steps_taken = 0
        self.prev_match_ratio = self._compute_match_ratio()

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

        # Guard against stale action tables (e.g. before load_task())
        structured_spec = None
        if isinstance(action, (list, tuple, np.ndarray)) and len(action) == self.action_component_dim:
            structured_spec = self._structured_action_to_spec(np.asarray(action, dtype=np.int64))

        if structured_spec is None:
            if not self._action_table:
                self._rebuild_action_table()
            action = int(np.clip(action, 0, len(self._action_table) - 1))
            spec = self._action_table[action]
        else:
            spec = structured_spec
        prev_match = self._compute_match_ratio()

        reward = 0.0
        done = False

        op_type = spec['type']

        if op_type == 'paint_cell':
            self._apply_paint_cell(spec)
        elif op_type == 'fill_region':
            self._apply_fill_region(spec)
        elif op_type == 'copy_patch':
            self._apply_copy_patch(spec)
        elif op_type == 'copy_exemplar_patch':
            self._apply_copy_exemplar_patch(spec)
        elif op_type == 'stamp_exemplar_output':
            self._apply_stamp_exemplar_output(spec)
        elif op_type == 'mirror':
            self._apply_mirror(spec)
        elif op_type == 'rotate':
            self._apply_rotate(spec)
        elif op_type == 'submit':
            done = True
        else:
            raise ValueError(f"Unknown action spec: {spec}")

        new_match = self._compute_match_ratio()
        match_gain = new_match - prev_match

        if op_type == 'submit':
            if self._is_solution_correct():
                reward = 1.0
            else:
                # Mild penalty for premature submission to encourage patience
                reward -= 0.1
        elif self._is_solution_correct():
            done = True

        if self.steps_taken >= self.max_steps and not done:
            done = True

        self.prev_match_ratio = new_match

        # Update state
        self.current_state = self._create_state_observation()

        info = {
            'steps_taken': self.steps_taken,
            'grid_match_ratio': new_match,
            'match_gain': match_gain,
            'action_type': op_type,
        }

        return self.current_state.copy(), reward, done, info

    def _spec_key(self, spec: Dict[str, Any]) -> Tuple[Any, ...]:
        op_type = spec['type']
        if op_type in {'paint_cell', 'fill_region'}:
            return (op_type, spec['row'], spec['col'], spec['color'])
        if op_type == 'copy_patch':
            return (
                op_type,
                spec['src_row'],
                spec['src_col'],
                spec['dst_row'],
                spec['dst_col'],
            )
        if op_type == 'copy_exemplar_patch':
            return (
                op_type,
                spec['example_idx'],
                spec.get('source', 'input'),
                spec['src_row'],
                spec['src_col'],
                spec['dst_row'],
                spec['dst_col'],
            )
        if op_type == 'stamp_exemplar_output':
            return (op_type, spec['example_idx'], spec['dst_row'], spec['dst_col'])
        if op_type == 'mirror':
            return (op_type, spec.get('axis', 'horizontal'))
        if op_type == 'rotate':
            return (op_type, spec.get('k', 0))
        if op_type == 'submit':
            return (op_type,)
        return (op_type,)

    def _snap_to_stride(self, value: int, stride: int, limit: int) -> int:
        if limit <= 0:
            return 0
        stride = max(1, stride)
        value = max(0, min(value, limit - 1))
        return (value // stride) * stride

    def _pointer_to_grid(self, pointer_idx: int, stride: int) -> Tuple[int, int]:
        pointer_idx = max(int(pointer_idx), 0)
        working = max(1, self.working_grid_size)
        row_cell = pointer_idx // working
        col_cell = pointer_idx % working
        if self.grid_height <= 1:
            row = 0
        else:
            row_ratio = row_cell / max(1, working - 1)
            row = int(np.clip(round(row_ratio * (self.grid_height - 1)), 0, self.grid_height - 1))
        if self.grid_width <= 1:
            col = 0
        else:
            col_ratio = col_cell / max(1, working - 1)
            col = int(np.clip(round(col_ratio * (self.grid_width - 1)), 0, self.grid_width - 1))

        snapped_row = self._snap_to_stride(row, stride, self.grid_height)
        snapped_col = self._snap_to_stride(col, stride, self.grid_width)
        return snapped_row, snapped_col

    def _structured_action_to_spec(self, action_vec: np.ndarray) -> Optional[Dict[str, Any]]:
        if action_vec.size < self.action_component_dim:
            return None

        operator_idx = int(np.clip(action_vec[0], 0, len(self.operator_names) - 1))
        op_type = self.operator_names[operator_idx]
        spec: Dict[str, Any] = {'type': op_type}

        if op_type in {'paint_cell', 'fill_region'}:
            pointer = int(action_vec[2]) if action_vec[2] >= 0 else 0
            color = int(np.clip(action_vec[3], 0, self.NUM_COLORS - 1))
            row, col = self._pointer_to_grid(pointer, self.action_stride)
            spec.update({'row': row, 'col': col, 'color': color})
        elif op_type == 'mirror':
            orientation = int(action_vec[5]) if action_vec[5] >= 0 else 0
            spec['axis'] = 'horizontal' if orientation % 2 == 0 else 'vertical'
        elif op_type == 'rotate':
            orientation = int(action_vec[5]) if action_vec[5] >= 0 else 0
            spec['k'] = (orientation % 3) + 1
        elif op_type == 'submit':
            pass
        else:
            return None

        return spec

    def action_index_from_structured(self, action_vec: np.ndarray) -> Optional[int]:
        spec = self._structured_action_to_spec(action_vec)
        if spec is None:
            return None
        key = self._spec_key(spec)
        return self._structured_lookup.get(key)

    def _create_state_observation(self) -> np.ndarray:
        """
        Create state observation from current grid and task context.

        Returns:
            state: Flattened state vector
        """
        # Flatten current grid to MAX_GRID_SIZE x MAX_GRID_SIZE (pad with zeros)
        grid_flat, downsampled_grid = self._pad_and_normalize_grid(self.current_grid)

        spatial_features = self._extract_spatial_features(downsampled_grid)
        task_features = self._cached_task_features

        # Metadata context (normalized for stability)
        exemplar_ratio = (
            min(len(self.training_examples), self.max_training_examples)
            / max(1, self.max_training_examples)
        )
        match_feature = self.prev_match_ratio if self.mode == "train" else 0.0
        entropy_val = self._compute_entropy_feature(downsampled_grid)
        object_fraction = self._compute_object_fraction(downsampled_grid)
        symmetry_score = self._compute_symmetry_score(downsampled_grid)
        context = np.array([
            self.grid_height / self.MAX_GRID_SIZE,
            self.grid_width / self.MAX_GRID_SIZE,
            exemplar_ratio,
            1.0 - (self.steps_taken / max(1, self.max_steps)),
            match_feature,
            entropy_val,
            object_fraction,
            symmetry_score,
        ], dtype=np.float32)

        # Concatenate
        state = np.concatenate([grid_flat, spatial_features, task_features, context])
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

    def _compute_task_context_features(self) -> np.ndarray:
        """Encode training exemplars into a fixed-size embedding.

        The encoding captures color distributions and coarse structural
        properties for up to ``MAX_TRAINING_EXAMPLES`` exemplar pairs. This
        gives the policy and option models a task-conditioned signal instead of
        averaging over unrelated puzzles.
        """

        features: List[float] = []
        for idx in range(self.max_training_examples):
            if idx < len(self.training_examples):
                input_grid, output_grid = self.training_examples[idx]
                features.extend(self._summarize_grid(input_grid))
                features.extend(self._summarize_grid(output_grid))
                features.append(self._grid_histogram_distance(input_grid, output_grid))
                features.append(input_grid.shape[0] / self.MAX_GRID_SIZE)
                features.append(input_grid.shape[1] / self.MAX_GRID_SIZE)
            else:
                features.extend([0.0] * (2 * self.NUM_COLORS + 3))

        return np.array(features, dtype=np.float32)

    def _pad_and_normalize_grid(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return normalized grid resized to working size and flattened."""

        target = np.zeros((self.working_grid_size, self.working_grid_size), dtype=np.float32)
        if grid.size > 0:
            normalized = grid.astype(np.float32) / (self.NUM_COLORS - 1)
            resized = self._resize_to_working_grid(normalized)
            h, w = resized.shape
            target[:h, :w] = resized
            return target.flatten(), resized

        return target.flatten(), target

    def _resize_to_working_grid(self, grid: np.ndarray) -> np.ndarray:
        """Downsample grid to working_grid_size using average pooling."""

        target = self.working_grid_size
        if grid.shape == (target, target):
            return grid

        h, w = grid.shape
        row_edges = np.linspace(0, h, num=target + 1, dtype=int)
        col_edges = np.linspace(0, w, num=target + 1, dtype=int)
        downsampled = np.zeros((target, target), dtype=np.float32)

        for i in range(target):
            r0, r1 = row_edges[i], row_edges[i + 1]
            if r0 == r1:
                r1 = min(r0 + 1, h)
            for j in range(target):
                c0, c1 = col_edges[j], col_edges[j + 1]
                if c0 == c1:
                    c1 = min(c0 + 1, w)
                window = grid[r0:r1, c0:c1]
                downsampled[i, j] = float(window.mean()) if window.size else 0.0

        return downsampled

    def _summarize_grid(self, grid: np.ndarray) -> List[float]:
        """Return normalized color histogram for a grid."""

        counts = np.bincount(grid.flatten(), minlength=self.NUM_COLORS).astype(np.float32)
        histogram = counts / max(1.0, counts.sum())
        return histogram.tolist()

    def _grid_histogram_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute L1 distance between two color histograms."""

        hist_a = np.bincount(a.flatten(), minlength=self.NUM_COLORS).astype(np.float32)
        hist_b = np.bincount(b.flatten(), minlength=self.NUM_COLORS).astype(np.float32)
        hist_a /= max(1.0, hist_a.sum())
        hist_b /= max(1.0, hist_b.sum())
        return float(np.abs(hist_a - hist_b).sum())

    def _compute_entropy_feature(self, normalized_grid: np.ndarray) -> float:
        if normalized_grid.size == 0:
            return 0.0
        colors = np.clip(np.round(normalized_grid * (self.NUM_COLORS - 1)), 0, self.NUM_COLORS - 1).astype(int)
        hist = np.bincount(colors.flatten(), minlength=self.NUM_COLORS).astype(np.float32)
        probs = hist / max(1.0, hist.sum())
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return float(min(entropy / math.log(self.NUM_COLORS), 1.0))

    def _compute_object_fraction(self, normalized_grid: np.ndarray) -> float:
        if normalized_grid.size == 0:
            return 0.0
        return float(np.count_nonzero(normalized_grid > 0.05) / normalized_grid.size)

    def _compute_symmetry_score(self, normalized_grid: np.ndarray) -> float:
        if normalized_grid.size == 0:
            return 0.0
        h_sym = np.mean(normalized_grid == np.fliplr(normalized_grid))
        v_sym = np.mean(normalized_grid == np.flipud(normalized_grid))
        return float((h_sym + v_sym) / 2.0)

    def _extract_spatial_features(self, grid: np.ndarray) -> np.ndarray:
        """Simple handcrafted spatial backbone.

        We avoid heavy ML frameworks so the feature extractor is implemented in
        NumPy. It applies a bank of small convolutional filters and records
        global statistics. This gives downstream models access to local
        structure (edges, blobs) that is completely missing when using only the
        flattened grid.
        """

        if grid.size == 0:
            dim = getattr(self, '_spatial_feature_dim', 19)
            return np.zeros(dim, dtype=np.float32)

        kernels = [
            np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32),  # vertical edge
            np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32),  # horizontal edge
            np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),  # laplacian
            np.ones((3, 3), dtype=np.float32) / 9.0,  # smoothing / blob detector
        ]

        feats: List[float] = []
        padded = np.pad(grid, 1, mode='edge')

        for kernel in kernels:
            conv = self._convolve_valid(padded, kernel)
            feats.append(conv.mean())
            feats.append(np.abs(conv).mean())
            feats.append(conv.max())
            feats.append(conv.min())

        # Global summary statistics
        feats.append(grid.mean())
        feats.append(grid.std())
        feats.append(float(np.count_nonzero(grid >= 0.5)) / grid.size)

        return np.array(feats, dtype=np.float32)

    def _convolve_valid(self, padded: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Perform a valid convolution of ``kernel`` over ``padded``."""

        kh, kw = kernel.shape
        out_h = padded.shape[0] - kh + 1
        out_w = padded.shape[1] - kw + 1
        output = np.zeros((out_h, out_w), dtype=np.float32)

        for i in range(out_h):
            for j in range(out_w):
                window = padded[i : i + kh, j : j + kw]
                output[i, j] = float(np.sum(window * kernel))

        return output

    def _rebuild_action_table(self):
        """Recompute the discrete action lookup table.

        The resulting space contains structured grid operations:
        * paint individual cells to specific colours
        * flood-fill connected regions with a target colour
        * copy small patches between coarse locations
        * mirror / rotate global patterns
        * submit the current hypothesis
        """
        height = max(1, self.grid_height)
        width = max(1, self.grid_width)

        stride = max(1, self.action_stride)
        copy_stride = max(stride, self.COPY_PATCH_STRIDE)

        action_table: List[Dict[str, Any]] = []

        # Paint actions
        paint_actions: List[Dict[str, Any]] = []
        for r in range(0, height, stride):
            for c in range(0, width, stride):
                for color in range(self.NUM_COLORS):
                    paint_actions.append({
                        'type': 'paint_cell',
                        'row': r,
                        'col': c,
                        'color': color,
                    })
        random.shuffle(paint_actions)
        if self.max_paint_actions and len(paint_actions) > self.max_paint_actions:
            paint_actions = paint_actions[: self.max_paint_actions]
        action_table.extend(paint_actions)

        # Flood-fill actions
        fill_actions: List[Dict[str, Any]] = []
        for r in range(0, height, stride):
            for c in range(0, width, stride):
                for color in range(self.NUM_COLORS):
                    fill_actions.append({
                        'type': 'fill_region',
                        'row': r,
                        'col': c,
                        'color': color,
                    })
        random.shuffle(fill_actions)
        if self.max_fill_actions and len(fill_actions) > self.max_fill_actions:
            fill_actions = fill_actions[: self.max_fill_actions]
        action_table.extend(fill_actions)

        # In-grid copy actions
        copy_actions: List[Dict[str, Any]] = []
        for src_r in range(0, height, copy_stride):
            for src_c in range(0, width, copy_stride):
                for dst_r in range(0, height, copy_stride):
                    for dst_c in range(0, width, copy_stride):
                        if src_r == dst_r and src_c == dst_c:
                            continue
                        copy_actions.append({
                            'type': 'copy_patch',
                            'src_row': src_r,
                            'src_col': src_c,
                            'dst_row': dst_r,
                            'dst_col': dst_c,
                        })
        random.shuffle(copy_actions)
        if self.max_copy_actions and len(copy_actions) > self.max_copy_actions:
            copy_actions = copy_actions[: self.max_copy_actions]
        action_table.extend(copy_actions)

        # Exemplar-based actions
        exemplar_actions: List[Dict[str, Any]] = []
        exemplar_cap = min(len(self.training_examples), self.max_training_examples)
        exemplar_stride = max(copy_stride, stride)
        for ex_idx in range(exemplar_cap):
            input_grid, output_grid = self.training_examples[ex_idx]

            for source_name, source_grid in (("input", input_grid), ("output", output_grid)):
                src_h, src_w = source_grid.shape
                for src_r in range(0, src_h, exemplar_stride):
                    for src_c in range(0, src_w, exemplar_stride):
                        for dst_r in range(0, height, exemplar_stride):
                            for dst_c in range(0, width, exemplar_stride):
                                exemplar_actions.append({
                                    'type': 'copy_exemplar_patch',
                                    'example_idx': ex_idx,
                                    'source': source_name,
                                    'src_row': src_r,
                                    'src_col': src_c,
                                    'dst_row': dst_r,
                                    'dst_col': dst_c,
                                })

            out_h, out_w = output_grid.shape
            for dst_r in range(0, height, exemplar_stride):
                for dst_c in range(0, width, exemplar_stride):
                    exemplar_actions.append({
                        'type': 'stamp_exemplar_output',
                        'example_idx': ex_idx,
                        'dst_row': dst_r,
                        'dst_col': dst_c,
                    })

        random.shuffle(exemplar_actions)

        if self.max_exemplar_actions and len(exemplar_actions) > self.max_exemplar_actions:
            exemplar_actions = exemplar_actions[: self.max_exemplar_actions]
        action_table.extend(exemplar_actions)

        action_table.extend([
            {'type': 'mirror', 'axis': 'horizontal'},
            {'type': 'mirror', 'axis': 'vertical'},
            {'type': 'rotate', 'k': 1},
            {'type': 'rotate', 'k': 2},
            {'type': 'rotate', 'k': 3},
            {'type': 'submit'},
        ])

        self._action_table = action_table
        self._action_dim = len(action_table)
        self._structured_lookup = {}
        for idx, spec in enumerate(action_table):
            self._structured_lookup[self._spec_key(spec)] = idx

    def _apply_paint_cell(self, spec: Dict[str, Any]):
        row = spec['row']
        col = spec['col']
        color = spec['color']

        if row < self.grid_height and col < self.grid_width:
            self.current_grid[row, col] = color

    def _apply_fill_region(self, spec: Dict[str, Any]):
        row = spec['row']
        col = spec['col']
        color = spec['color']

        if row >= self.grid_height or col >= self.grid_width:
            return

        target_color = self.current_grid[row, col]
        if target_color == color:
            return

        stack = [(row, col)]
        visited = set()

        while stack:
            r, c = stack.pop()
            if (
                r < 0
                or r >= self.grid_height
                or c < 0
                or c >= self.grid_width
                or (r, c) in visited
                or self.current_grid[r, c] != target_color
            ):
                continue

            visited.add((r, c))
            self.current_grid[r, c] = color
            stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])

    def _apply_copy_patch(self, spec: Dict[str, Any]):
        src_r = spec['src_row']
        src_c = spec['src_col']
        dst_r = spec['dst_row']
        dst_c = spec['dst_col']

        patch = self._extract_patch(src_r, src_c)
        if patch.size == 0:
            return

        h, w = patch.shape
        end_r = min(dst_r + h, self.grid_height)
        end_c = min(dst_c + w, self.grid_width)
        patch_h = end_r - dst_r
        patch_w = end_c - dst_c

        if patch_h <= 0 or patch_w <= 0:
            return

        self.current_grid[dst_r:end_r, dst_c:end_c] = patch[:patch_h, :patch_w]

    def _apply_copy_exemplar_patch(self, spec: Dict[str, Any]):
        example_idx = spec['example_idx']
        source = spec['source']
        src_r = spec['src_row']
        src_c = spec['src_col']
        dst_r = spec['dst_row']
        dst_c = spec['dst_col']

        patch = self._extract_exemplar_patch(example_idx, source, src_r, src_c)
        if patch.size == 0:
            return

        h, w = patch.shape
        end_r = min(dst_r + h, self.grid_height)
        end_c = min(dst_c + w, self.grid_width)
        patch_h = end_r - dst_r
        patch_w = end_c - dst_c

        if patch_h <= 0 or patch_w <= 0:
            return

        self.current_grid[dst_r:end_r, dst_c:end_c] = patch[:patch_h, :patch_w]

    def _apply_stamp_exemplar_output(self, spec: Dict[str, Any]):
        example_idx = spec['example_idx']
        dst_r = spec['dst_row']
        dst_c = spec['dst_col']

        exemplar = self._get_exemplar_grid(example_idx, 'output')
        if exemplar is None:
            return

        h, w = exemplar.shape
        end_r = min(dst_r + h, self.grid_height)
        end_c = min(dst_c + w, self.grid_width)
        if end_r <= dst_r or end_c <= dst_c:
            return

        self.current_grid[dst_r:end_r, dst_c:end_c] = exemplar[: end_r - dst_r, : end_c - dst_c]

    def _extract_patch(self, row: int, col: int) -> np.ndarray:
        end_r = min(row + self.COPY_PATCH_SIZE, self.grid_height)
        end_c = min(col + self.COPY_PATCH_SIZE, self.grid_width)
        if end_r <= row or end_c <= col:
            return np.array([], dtype=self.current_grid.dtype)
        return self.current_grid[row:end_r, col:end_c].copy()

    def _get_exemplar_grid(self, index: int, source: str) -> Optional[np.ndarray]:
        if not (0 <= index < len(self.training_examples)):
            return None

        if source == 'input':
            return self.training_examples[index][0]
        if source == 'output':
            return self.training_examples[index][1]
        return None

    def _extract_exemplar_patch(self, index: int, source: str, row: int, col: int) -> np.ndarray:
        grid = self._get_exemplar_grid(index, source)
        if grid is None:
            return np.array([], dtype=np.int32)

        end_r = min(row + self.COPY_PATCH_SIZE, grid.shape[0])
        end_c = min(col + self.COPY_PATCH_SIZE, grid.shape[1])
        if end_r <= row or end_c <= col:
            return np.array([], dtype=grid.dtype)

        return grid[row:end_r, col:end_c]

    def _apply_mirror(self, spec: Dict[str, Any]):
        axis = spec['axis']
        if axis == 'horizontal':
            self.current_grid = np.fliplr(self.current_grid)
        elif axis == 'vertical':
            self.current_grid = np.flipud(self.current_grid)

    def _apply_rotate(self, spec: Dict[str, Any]):
        k = spec['k'] % 4
        if k:
            original = self.current_grid
            rotated = np.rot90(original, k=k)
            target = np.zeros_like(original)
            h = min(target.shape[0], rotated.shape[0])
            w = min(target.shape[1], rotated.shape[1])
            target[:h, :w] = rotated[:h, :w]
            self.current_grid = target

    def get_state_components(self, state: Optional[np.ndarray] = None) -> Tuple:
        """
        Extract meaningful components from state for GVF cumulants.

        Returns:
            (grid_entropy, object_count, symmetry_score, match_ratio)
        """
        if state is None or len(state) < self._grid_flat_dim:
            grid = self.current_grid
        else:
            flat_grid = state[:self._grid_flat_dim]
            grid = flat_grid.reshape(self.working_grid_size, self.working_grid_size)
            grid = np.round(grid * (self.NUM_COLORS - 1)).astype(int)

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

    def get_feature_layout(self) -> Dict[str, int]:
        """Return lengths of each feature segment in ARC observation."""

        return {
            'grid_flat': self._grid_flat_dim,
            'spatial': self._spatial_feature_dim,
            'task': self._task_feature_dim,
            'context': self._context_dim,
            'working_grid_size': self.working_grid_size,
            'max_training_examples': self.max_training_examples,
        }

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
