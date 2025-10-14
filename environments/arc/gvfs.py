"""
ARC-Specific GVF Definitions
Predictive knowledge layer for ARC domain

GVFs predict grid properties that guide option discovery:
- Grid similarity to solution
- Grid entropy (complexity)
- Object count
- Symmetry
- Transformation progress
"""
import math
import numpy as np
from knowledge.gvf import GVF


class GridSimilarityGVF(GVF):
    """g1: Predicts E[similarity to solution] - how close grid is to target"""

    def __init__(self, state_dim, hidden_size=128, gamma=0.95):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "grid_similarity"

    def compute_cumulant(self, state):
        """
        Cumulant: Ratio of cells matching solution ∈ [0, 1]

        Note: This requires access to environment for match_ratio.
        In practice, match_ratio should be part of the state or computed from state.
        For now, we assume it's encoded in the state or available via info dict.
        """
        # Extract match ratio from state components (last few dimensions)
        # This is a simplified implementation - in practice, would need proper state encoding
        if isinstance(state, np.ndarray) and len(state) > 3:
            # Assume match ratio is encoded or we compute from grid
            # Placeholder: return dummy value for now
            return 0.5  # TODO: Extract actual similarity from state
        return 0.5


class EntropyGVF(GVF):
    """g2: Predicts E[grid entropy] - complexity/disorder of grid"""

    def __init__(self, state_dim, hidden_size=128, gamma=0.95):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "entropy"
        self.max_entropy = math.log(10)  # Maximum entropy for 10 colors

    def compute_cumulant(self, state):
        """
        Cumulant: Shannon entropy of grid normalized ∈ [0, 1]

        Grid entropy indicates complexity - higher entropy = more diverse colors
        """
        if isinstance(state, np.ndarray):
            # Extract grid from state (first 900 elements for 30x30 grid)
            grid_size = 30 * 30
            if len(state) >= grid_size:
                grid = state[:grid_size].reshape(30, 30)

                # Compute entropy
                unique, counts = np.unique(grid, return_counts=True)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-8))

                # Normalize by max entropy
                return min(entropy / self.max_entropy, 1.0)

        return 0.5  # Default


class ObjectCountGVF(GVF):
    """g3: Predicts E[num_objects] - number of distinct objects/regions"""

    def __init__(self, state_dim, hidden_size=128, gamma=0.95):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "object_count"
        self.max_objects = 50  # Assume max 50 objects in a grid

    def compute_cumulant(self, state):
        """
        Cumulant: Number of non-background cells normalized ∈ [0, 1]

        Simplified: Count non-zero cells as proxy for object presence
        """
        if isinstance(state, np.ndarray):
            # Extract grid from state
            grid_size = 30 * 30
            if len(state) >= grid_size:
                grid = state[:grid_size].reshape(30, 30)

                # Count non-zero cells
                object_cells = np.sum(grid > 0)

                # Normalize by max possible objects
                return min(object_cells / (self.max_objects * 4), 1.0)  # *4 for cell size

        return 0.5  # Default


class SymmetryGVF(GVF):
    """g4: Predicts E[symmetry] - horizontal/vertical symmetry of grid"""

    def __init__(self, state_dim, hidden_size=128, gamma=0.95):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "symmetry"

    def compute_cumulant(self, state):
        """
        Cumulant: Average of horizontal and vertical symmetry ∈ [0, 1]

        Symmetry = fraction of cells that match after mirroring
        """
        if isinstance(state, np.ndarray):
            # Extract grid from state
            grid_size = 30 * 30
            if len(state) >= grid_size:
                grid = state[:grid_size].reshape(30, 30)

                # Compute horizontal symmetry
                h_sym = np.mean(grid == np.fliplr(grid))

                # Compute vertical symmetry
                v_sym = np.mean(grid == np.flipud(grid))

                # Average symmetry
                return (h_sym + v_sym) / 2.0

        return 0.5  # Default


class ProgressGVF(GVF):
    """g5: Predicts E[task progress] - how far along the transformation we are"""

    def __init__(self, state_dim, hidden_size=128, gamma=0.95):
        super().__init__(state_dim, hidden_size, gamma)
        self.name = "progress"

    def compute_cumulant(self, state):
        """
        Cumulant: Task completion progress ∈ [0, 1]

        Progress is measured by similarity to solution (if available)
        This is similar to GridSimilarityGVF but with different γ for longer-term prediction
        """
        # In practice, this would be computed from state or environment info
        # Placeholder implementation
        return 0.5


class ARCHordeGVFs:
    """
    Horde of GVFs for ARC - collection of all predictive knowledge
    Updates all GVFs in parallel (OaK continual learning)
    """

    def __init__(self, state_dim, config, meta_config=None):
        self.state_dim = state_dim

        # Create the 5 core GVFs for ARC
        gvf_hidden = getattr(config, 'GVF_HIDDEN_SIZE', 128)
        gvf_gamma = getattr(config, 'GVF_GAMMA', 0.95)

        self.gvfs = {
            'grid_similarity': GridSimilarityGVF(state_dim, gvf_hidden, gvf_gamma),
            'entropy': EntropyGVF(state_dim, gvf_hidden, gvf_gamma),
            'object_count': ObjectCountGVF(state_dim, gvf_hidden, gvf_gamma),
            'symmetry': SymmetryGVF(state_dim, gvf_hidden, gvf_gamma),
            'progress': ProgressGVF(state_dim, gvf_hidden, 0.99),  # Longer horizon for progress
        }

        self.config = config

        base_lr = getattr(config, 'GVF_LR', 1e-3)
        meta_cfg = None
        if meta_config is not None:
            meta_cfg = meta_config.copy()
            meta_cfg['init_log_alpha'] = math.log(base_lr)
        for gvf in self.gvfs.values():
            gvf.configure_optimizer(base_lr=base_lr, meta_config=meta_cfg)

    def predict_all(self, state):
        """Get predictions from all GVFs"""
        return {name: gvf.predict(state) for name, gvf in self.gvfs.items()}

    def update_all(self, state, next_state, done=False, step_sizes=None):
        """
        Update all GVFs with their respective cumulants

        Returns: dict of td_errors for meta-learning
        """
        td_errors = {}

        for name, gvf in self.gvfs.items():
            # Compute cumulant for this GVF
            cumulant = gvf.compute_cumulant(state)

            # Get step-size from IDBD if provided
            step_size = step_sizes.get(name) if step_sizes else None

            # Update GVF
            gamma = 0.0 if done else gvf.gamma
            td_error = gvf.update(state, cumulant, next_state, gamma=gamma, step_size=step_size)
            td_errors[name] = td_error

        return td_errors

    def get_gvf(self, name):
        """Get a specific GVF by name"""
        return self.gvfs.get(name)

    def get_all_gvfs(self):
        """Get all GVFs"""
        return self.gvfs

    def get_average_errors(self):
        """Get average prediction error for each GVF"""
        return {name: gvf.get_average_error() for name, gvf in self.gvfs.items()}

    def get_normalized_errors(self):
        """Get normalized TD errors for each GVF"""
        return {name: gvf.get_normalized_error() for name, gvf in self.gvfs.items()}

    def __getitem__(self, name):
        """Access GVF by name"""
        return self.gvfs[name]
