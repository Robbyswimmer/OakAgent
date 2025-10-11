"""
Meta-Learning Optimizer Adapter

Provides adaptive per-parameter learning rates via IDBD (Incremental Delta-Bar-Delta).
Implements Sutton (1992) for online second-order optimization in continual learning.

This module enables switching between:
- Fixed learning rates (Adam) for baseline comparison
- Adaptive learning rates (IDBD/TIDBD/Autostep) for continual learning

Reference:
    Sutton, R.S. (1992). Adapting bias by gradient descent: An incremental
    version of delta-bar-delta. AAAI, 171-176.
"""
import numpy as np
import torch

from .idbd import TorchIDBD


class MetaOptimizerAdapter:
    """
    Optimizer wrapper supporting both fixed (Adam) and adaptive (IDBD) learning rates.

    IDBD (Sutton, 1992) adjusts each parameter's learning rate α_i based on
    gradient consistency. This enables:
    - Faster learning when gradients are consistent (same direction)
    - Slower learning when gradients oscillate (prevents instability)
    - Adaptation to non-stationarity (critical for continual learning)

    Mathematical Update:
    -------------------
    For each parameter θ_i at time t:

        θ_i(t+1) = θ_i(t) - α_i(t) · ∇_i(t)              [parameter update]
        log(α_i(t+1)) = log(α_i(t)) + μ · ∇_i(t) · h_i(t)  [meta-update]
        h_i(t+1) = β·h_i(t) + (1-β)·∇_i(t)               [eligibility trace]

    Where:
        α_i : Per-parameter learning rate (adaptive)
        μ   : Meta-learning rate (how fast α adapts), typically 3e-4
        h_i : Exponentially weighted gradient history (eligibility trace)
        β   : Trace decay factor, typically 0.99
        ∇_i : Gradient ∂L/∂θ_i

    Intuition:
    ----------
    Think of h_i as a "momentum" of gradient direction. When the current
    gradient ∇_i aligns with momentum h_i (dot product > 0), we're moving
    consistently → increase α_i to accelerate learning. When they point in
    opposite directions (oscillation) → decrease α_i to stabilize.

    This provides automatic:
    - Learning rate annealing (α decreases as convergence approached)
    - Acceleration in high-gradient regions (α increases when gradients consistent)
    - Robustness to non-stationarity (adapts to distribution shift)

    Parameters:
    -----------
    parameters : iterable of torch.nn.Parameter
        Model parameters to optimize (e.g., network.parameters())

    base_lr : float, default=1e-3
        Initial learning rate. Used for:
        - Adam's fixed rate (when meta_config=None)
        - IDBD's initial α for all parameters (when using meta-learning)

    meta_config : dict or None
        If None: Use Adam with fixed learning rate (baseline)
        If dict: Use IDBD with adaptive rates (OaK compliance)

        Required keys when using IDBD:
            'mu' (float): Meta-learning rate, typically 3e-4
                Controls how fast α adapts. Higher = faster adaptation
                but risk of instability. Lower = slower but more stable.

            'init_log_alpha' (float): log(initial α), typically log(base_lr)
                All parameters start with same learning rate.

            'type' (str): 'idbd', 'tidbd', or 'autostep'
                - 'idbd': Original Sutton (1992) algorithm
                - 'tidbd': Uses temporal difference of gradients (Kearney et al. 2019)
                - 'autostep': Adds safety mechanisms (Mahmood et al. 2012)

            'min_alpha' (float): Lower bound on α, typically 1e-6
                Prevents complete learning shutdown.

            'max_alpha' (float): Upper bound on α, typically 0.1
                Prevents runaway growth and instability.

            'disabled' (bool): If True, fall back to Adam even if config provided

    Attributes:
    -----------
    use_meta : bool
        True if using IDBD, False if using Adam

    meta_updater : TorchIDBD or None
        IDBD implementation (only when use_meta=True)

    optimizer : torch.optim.Adam or None
        Adam optimizer (only when use_meta=False)

    parameters : list
        List of parameters being optimized

    base_lr : float
        Base/initial learning rate

    Methods:
    --------
    zero_grad()
        Clear all parameter gradients (call before loss.backward())

    step(feature_vector, clip_range)
        Apply one optimization step (call after loss.backward())
        Requires gradients to be computed first.

    get_average_step_size()
        Get mean α across all parameters (diagnostic/logging)

    Example Usage:
    --------------
    # With IDBD (adaptive learning rates):
    >>> meta_config = {
    ...     'mu': 3e-4,                    # Meta-learning rate
    ...     'init_log_alpha': np.log(1e-3), # Initial log(α)
    ...     'type': 'idbd',                 # Algorithm variant
    ...     'min_alpha': 1e-6,              # Minimum α
    ...     'max_alpha': 0.1                # Maximum α
    ... }
    >>> optimizer = MetaOptimizerAdapter(
    ...     model.parameters(),
    ...     base_lr=1e-3,
    ...     meta_config=meta_config
    ... )
    >>>
    >>> # Training loop:
    >>> for state, action, reward, next_state in data:
    ...     loss = compute_loss(state, action, reward, next_state)
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step(feature_vector=state, clip_range=(-10, 10))

    # With Adam (fixed learning rate, baseline):
    >>> optimizer = MetaOptimizerAdapter(
    ...     model.parameters(),
    ...     base_lr=1e-3,
    ...     meta_config=None  # Use Adam
    ... )
    >>>
    >>> # Training loop (simpler):
    >>> for batch in data:
    ...     loss = compute_loss(batch)
    ...     optimizer.zero_grad()
    ...     loss.backward()
    ...     optimizer.step()  # No feature_vector needed for Adam

    Notes:
    ------
    - IDBD is particularly valuable in continual learning where the
      data distribution shifts over time (non-stationarity). Fixed
      learning rates may be too fast initially or too slow later.

    - The feature_vector parameter allows state-dependent adaptation:
      learning rates can vary based on which state region we're in.
      This is useful for environments where some states are rare and
      need more aggressive learning.

    - Gradient clipping (clip_range) is strongly recommended to prevent
      runaway updates during early training when α might be large.

    - For OaK compliance, IDBD should be enabled (meta_config provided).
      Adam mode is provided for ablation studies to demonstrate the
      value of adaptive learning rates.

    See Also:
    ---------
    TorchIDBD : Underlying IDBD implementation for PyTorch
    IDBD, TIDBD, Autostep : Core meta-learning algorithms

    References:
    -----------
    See REFERENCES.md in the repository root for complete citations.
    """

    def __init__(self, parameters, base_lr=1e-3, meta_config=None):
        self.parameters = list(parameters)
        self.base_lr = base_lr
        self.use_meta = meta_config is not None and not meta_config.get('disabled', False)

        if self.use_meta:
            init_log_alpha = meta_config.get('init_log_alpha', np.log(base_lr))
            self.meta_updater = TorchIDBD(
                self.parameters,
                mu=meta_config.get('mu', 1e-3),
                init_log_alpha=init_log_alpha,
                meta_type=meta_config.get('type', 'idbd'),
                min_alpha=meta_config.get('min_alpha', 1e-6),
                max_alpha=meta_config.get('max_alpha', 1.0),
            )
            self.optimizer = None
        else:
            self.optimizer = torch.optim.Adam(self.parameters, lr=base_lr)
            self.meta_updater = None

    def zero_grad(self):
        if self.use_meta:
            for param in self.parameters:
                if param.grad is not None:
                    param.grad.zero_()
        else:
            self.optimizer.zero_grad(set_to_none=True)

    def step(self, feature_vector=None, clip_range=None):
        """Apply an optimization step. Gradients must already be computed."""
        if self.use_meta:
            grads = [param.grad for param in self.parameters]
            step_sizes = self.meta_updater.update_step_sizes(grads, feature_vector)
            self.meta_updater.apply_updates(self.parameters, step_sizes)
            if clip_range is not None:
                low, high = clip_range
                for param in self.parameters:
                    param.data.clamp_(low, high)
            for param in self.parameters:
                param.grad = None
        else:
            self.optimizer.step()
            if clip_range is not None:
                low, high = clip_range
                for param in self.parameters:
                    param.data.clamp_(low, high)

    def get_average_step_size(self):
        if not self.use_meta:
            return self.base_lr
        step_sizes = self.meta_updater.get_step_sizes()
        return float(np.mean(step_sizes)) if step_sizes.size > 0 else 0.0

