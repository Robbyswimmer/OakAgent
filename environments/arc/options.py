# -*- coding: utf-8 -*-
"""
ARC-Specific Option Templates
Temporal abstractions for ARC domain (grid transformations)
"""
import numpy as np
from options.option import Option


def _context_dict(state_segment):
    if state_segment is None or len(state_segment) < 8:
        return {
            'height_ratio': 0.0,
            'width_ratio': 0.0,
            'exemplar_ratio': 0.0,
            'steps_remaining': 0.0,
            'match_ratio': 0.0,
            'entropy': 0.5,
            'object_fraction': 0.0,
            'symmetry': 0.0,
        }
    height_ratio, width_ratio, exemplar_ratio, steps_remaining, match_ratio, entropy, object_frac, symmetry = state_segment[-8:]
    return {
        'height_ratio': float(height_ratio),
        'width_ratio': float(width_ratio),
        'exemplar_ratio': float(exemplar_ratio),
        'steps_remaining': float(steps_remaining),
        'match_ratio': float(match_ratio),
        'entropy': float(entropy),
        'object_fraction': float(object_frac),
        'symmetry': float(symmetry),
    }


class ARCOptionMixin:
    """Utility mixin exposing exemplar-aligned state helpers."""

    CONTEXT_DIM = 8
    DEFAULT_GRID = 16
    DEFAULT_MAX_EXEMPLARS = 7
    NUM_COLORS = 10

    def __init__(
        self,
        state_dim: int,
        exemplar_idx: int = 0,
        max_training_examples: int = None,
        spatial_feature_dim: int = None,
    ):
        self.num_colors = self.NUM_COLORS
        self.exemplar_dim = 2 * self.num_colors + 3
        self.context_dim = self.CONTEXT_DIM
        self.working_grid_size = self._infer_grid_size(state_dim)
        self.grid_flat_dim = self.working_grid_size * self.working_grid_size
        if max_training_examples is not None and spatial_feature_dim is not None:
            self.max_training_examples = int(max_training_examples)
            self.spatial_feature_dim = int(spatial_feature_dim)
        else:
            remainder = state_dim - self.grid_flat_dim - self.context_dim
            inferred_examples = None
            for candidate in range(self.DEFAULT_MAX_EXEMPLARS, 0, -1):
                spatial_dim = remainder - candidate * self.exemplar_dim
                if spatial_dim >= 0:
                    inferred_examples = candidate
                    break
            if inferred_examples is None:
                inferred_examples = self.DEFAULT_MAX_EXEMPLARS
                spatial_dim = max(0, state_dim - self.grid_flat_dim - inferred_examples * self.exemplar_dim - self.context_dim)
            self.max_training_examples = inferred_examples
            self.spatial_feature_dim = int(spatial_dim)
        self.exemplar_idx = int(np.clip(exemplar_idx, 0, self.max_training_examples - 1))

    def _infer_grid_size(self, state_dim: int) -> int:
        for candidate in range(4, 33):
            spatial = candidate * candidate
            remainder = state_dim - spatial - self.context_dim
            if remainder >= 0 and remainder % self.exemplar_dim == 0:
                return candidate
        return self.DEFAULT_GRID

    def _ensure_state(self, state) -> np.ndarray:
        if isinstance(state, np.ndarray):
            return state
        if hasattr(state, 'detach'):
            return state.detach().cpu().numpy()
        return np.asarray(state, dtype=np.float32)

    def _exemplar_features(self, state, exemplar_idx: int) -> np.ndarray:
        state_np = self._ensure_state(state)
        start = self.grid_flat_dim + self.spatial_feature_dim + exemplar_idx * self.exemplar_dim
        end = start + self.exemplar_dim
        return state_np[start:end]

    def _exemplar_output_histogram(self, state, exemplar_idx: int) -> np.ndarray:
        features = self._exemplar_features(state, exemplar_idx)
        output_hist = features[self.num_colors : 2 * self.num_colors].astype(np.float32)
        total = float(output_hist.sum())
        if total > 0:
            output_hist /= total
        else:
            output_hist = np.full(self.num_colors, 1.0 / self.num_colors, dtype=np.float32)
        return output_hist

    def _current_histogram(self, state) -> np.ndarray:
        state_np = self._ensure_state(state)
        grid_flat = state_np[: self.grid_flat_dim]
        colors = np.clip(np.round(grid_flat * (self.num_colors - 1)), 0, self.num_colors - 1).astype(int)
        hist = np.bincount(colors, minlength=self.num_colors).astype(np.float32)
        total = float(hist.sum())
        if total > 0:
            hist /= total
        else:
            hist = np.full(self.num_colors, 1.0 / self.num_colors, dtype=np.float32)
        return hist

    def _color_similarity(self, state, exemplar_idx: int) -> float:
        current = self._current_histogram(state)
        target = self._exemplar_output_histogram(state, exemplar_idx)
        return float(1.0 - 0.5 * np.abs(current - target).sum())

    def _target_entropy(self, state, exemplar_idx: int) -> float:
        target_hist = self._exemplar_output_histogram(state, exemplar_idx)
        entropy = -np.sum(target_hist * np.log(target_hist + 1e-8))
        return float(entropy / np.log(self.num_colors))

    def _context(self, state) -> dict:
        state_np = self._ensure_state(state)
        return _context_dict(state_np[-self.context_dim:])


class FillRegionOption(ARCOptionMixin, Option):
    """Fill a connected region so colors match exemplar distribution."""

    def __init__(self, option_id, name, state_dim, action_dim,
                 uniformity_threshold=0.9, exemplar_idx=0, max_length=15,
                 max_training_examples=None, spatial_feature_dim=None):
        ARCOptionMixin.__init__(
            self,
            state_dim,
            exemplar_idx=exemplar_idx,
            max_training_examples=max_training_examples,
            spatial_feature_dim=spatial_feature_dim,
        )
        Option.__init__(self, option_id, name, state_dim, action_dim, max_length)
        self.uniformity_threshold = uniformity_threshold

    def termination(self, state):
        similarity = self._color_similarity(state, self.exemplar_idx)
        achieved = similarity >= self.uniformity_threshold
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        similarity = self._color_similarity(state, self.exemplar_idx)
        return (similarity - self.uniformity_threshold) * 4.0


class SymmetrizeOption(ARCOptionMixin, Option):
    """Encourage symmetry while matching exemplar palette."""

    def __init__(self, option_id, name, state_dim, action_dim,
                 axis='horizontal', symmetry_threshold=0.8, exemplar_idx=0, max_length=20,
                 max_training_examples=None, spatial_feature_dim=None):
        ARCOptionMixin.__init__(
            self,
            state_dim,
            exemplar_idx=exemplar_idx,
            max_training_examples=max_training_examples,
            spatial_feature_dim=spatial_feature_dim,
        )
        Option.__init__(self, option_id, name, state_dim, action_dim, max_length)
        self.axis = axis
        self.symmetry_threshold = symmetry_threshold

    def termination(self, state):
        ctx = self._context(state)
        color_alignment = self._color_similarity(state, self.exemplar_idx)
        combined = 0.5 * (ctx['symmetry'] + color_alignment)
        achieved = combined >= self.symmetry_threshold
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        ctx = self._context(state)
        color_alignment = self._color_similarity(state, self.exemplar_idx)
        combined = 0.5 * (ctx['symmetry'] + color_alignment)
        distance = max(0.0, 1.0 - combined)
        reward = np.exp(-4.0 * distance) - 1.0
        if combined >= self.symmetry_threshold:
            reward += 1.0
        return reward * 2.0


class ReduceEntropyOption(ARCOptionMixin, Option):
    """Reduce entropy toward exemplar output distribution."""

    def __init__(self, option_id, name, state_dim, action_dim,
                 max_entropy=1.5, exemplar_idx=0, max_length=15,
                 max_training_examples=None, spatial_feature_dim=None):
        ARCOptionMixin.__init__(
            self,
            state_dim,
            exemplar_idx=exemplar_idx,
            max_training_examples=max_training_examples,
            spatial_feature_dim=spatial_feature_dim,
        )
        Option.__init__(self, option_id, name, state_dim, action_dim, max_length)
        self.base_entropy_norm = max_entropy / np.log(10)

    def termination(self, state):
        ctx = self._context(state)
        entropy = ctx['entropy']
        target_entropy = min(self.base_entropy_norm, self._target_entropy(state, self.exemplar_idx))
        achieved = entropy <= target_entropy
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        ctx = self._context(state)
        entropy = ctx['entropy']
        target_entropy = min(self.base_entropy_norm, self._target_entropy(state, self.exemplar_idx))
        return (target_entropy - entropy) * 2.5


class CopyPatternOption(ARCOptionMixin, Option):
    """Copy exemplar pattern measured via histogram alignment."""

    def __init__(self, option_id, name, state_dim, action_dim,
                 exemplar_idx=0, max_length=20,
                 max_training_examples=None, spatial_feature_dim=None):
        ARCOptionMixin.__init__(
            self,
            state_dim,
            exemplar_idx=exemplar_idx,
            max_training_examples=max_training_examples,
            spatial_feature_dim=spatial_feature_dim,
        )
        Option.__init__(self, option_id, name, state_dim, action_dim, max_length)

    def termination(self, state):
        similarity = self._color_similarity(state, self.exemplar_idx)
        achieved = similarity >= 0.6
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        similarity = self._color_similarity(state, self.exemplar_idx)
        return similarity - 0.5


class MatchSolutionOption(ARCOptionMixin, Option):
    """Drive grid to match exemplar solution distribution."""

    def __init__(self, option_id, name, state_dim, action_dim,
                 similarity_threshold=0.95, exemplar_idx=0, max_length=25,
                 max_training_examples=None, spatial_feature_dim=None):
        ARCOptionMixin.__init__(
            self,
            state_dim,
            exemplar_idx=exemplar_idx,
            max_training_examples=max_training_examples,
            spatial_feature_dim=spatial_feature_dim,
        )
        Option.__init__(self, option_id, name, state_dim, action_dim, max_length)
        self.similarity_threshold = similarity_threshold

    def termination(self, state):
        similarity = self._color_similarity(state, self.exemplar_idx)
        achieved = similarity >= self.similarity_threshold
        return achieved, achieved

    def compute_pseudo_reward(self, state):
        similarity = self._color_similarity(state, self.exemplar_idx)
        return similarity
