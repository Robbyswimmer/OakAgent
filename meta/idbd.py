"""
Meta-Learning: Incremental Delta-Bar-Delta (IDBD) and variants
Per-weight adaptive step-sizes for continual learning (OaK core principle #7)

References:
- Sutton (1992): IDBD
- TIDBD: Time-difference variant
- Autostep: Safety improvements
"""
import numpy as np
import torch
import torch.nn as nn

class IDBD:
    """
    Incremental Delta-Bar-Delta (IDBD)
    Learns per-weight step-sizes online using meta-gradient descent

    Each weight has:
    - log_alpha: log of step-size (learned)
    - h: eligibility trace for meta-learning
    """

    def __init__(self, num_weights, mu=1e-3, init_log_alpha=None, min_alpha=1e-6, max_alpha=1.0):
        """
        Args:
            num_weights: number of parameters to track
            mu: meta learning rate
            init_log_alpha: initial log(alpha) value
            min_alpha: minimum step-size
            max_alpha: maximum step-size
        """
        self.num_weights = num_weights
        self.mu = mu
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

        # Initialize log step-sizes
        if init_log_alpha is None:
            init_log_alpha = np.log(1e-3)
        self.log_alpha = np.ones(num_weights, dtype=np.float32) * init_log_alpha

        # Eligibility traces for meta-learning
        self.h = np.zeros(num_weights, dtype=np.float32)

        # Track previous gradients for meta-update
        self.prev_grad = np.zeros(num_weights, dtype=np.float32)

    def get_step_sizes(self):
        """Get current step-sizes (alpha = exp(log_alpha))"""
        alpha = np.exp(self.log_alpha)
        return np.clip(alpha, self.min_alpha, self.max_alpha)

    def update(self, gradient, features, td_error=None):
        """
        Update step-sizes using IDBD rule

        Args:
            gradient: current gradient (∂L/∂w)
            features: input features (for eligibility trace)
            td_error: TD error (optional, for TD-specific variants)

        Returns:
            updated step-sizes
        """
        # Update eligibility trace: h = h + features * gradient
        self.h = self.h + features * gradient

        # Meta-gradient: ∂log_alpha/∂t = μ * δ * h
        # where δ is the gradient (or TD error if provided)
        delta = td_error if td_error is not None else gradient

        # IDBD update
        meta_gradient = self.mu * delta * self.h
        self.log_alpha += meta_gradient

        # Decay eligibility traces
        self.h *= 0.99

        self.prev_grad = gradient

        return self.get_step_sizes()

    def reset_traces(self):
        """Reset eligibility traces (e.g., at episode boundaries if needed)"""
        self.h = np.zeros_like(self.h)


class TIDBD(IDBD):
    """
    Time-difference IDBD
    Uses temporal difference of gradients for meta-update
    """

    def update(self, gradient, features, td_error=None):
        """Update using time-difference of gradients"""
        # Compute gradient difference
        grad_diff = gradient - self.prev_grad

        # Update eligibility trace
        self.h = self.h + features * grad_diff

        # Meta-gradient using TD of gradients
        delta = td_error if td_error is not None else grad_diff
        meta_gradient = self.mu * delta * self.h
        self.log_alpha += meta_gradient

        # Decay
        self.h *= 0.99

        self.prev_grad = gradient

        return self.get_step_sizes()


class Autostep(IDBD):
    """
    Autostep: IDBD with additional safeguards
    - Normalized features
    - Bounded updates
    """

    def update(self, gradient, features, td_error=None):
        """Update with normalization and bounds"""
        # Normalize features
        feature_norm = np.linalg.norm(features) + 1e-8
        normalized_features = features / feature_norm

        # Update eligibility trace with normalized features
        self.h = self.h + normalized_features * gradient

        # Meta-gradient with TD error
        delta = td_error if td_error is not None else gradient
        meta_gradient = self.mu * delta * self.h

        # Clip meta-gradient for stability
        meta_gradient = np.clip(meta_gradient, -1.0, 1.0)

        self.log_alpha += meta_gradient

        # Decay
        self.h *= 0.99

        self.prev_grad = gradient

        return self.get_step_sizes()


class TorchIDBD(nn.Module):
    """
    PyTorch-compatible IDBD for neural network training
    Wraps optimizer and maintains per-parameter step-sizes
    """

    def __init__(self, parameters, mu=1e-3, init_log_alpha=None):
        super().__init__()
        self.mu = mu

        # Flatten all parameters
        self.param_shapes = []
        self.param_sizes = []
        total_params = 0

        for param in parameters:
            shape = param.shape
            size = param.numel()
            self.param_shapes.append(shape)
            self.param_sizes.append(size)
            total_params += size

        # Initialize IDBD for all parameters
        self.idbd = IDBD(total_params, mu=mu, init_log_alpha=init_log_alpha)

    def get_step_sizes(self):
        """Get step-sizes for all parameters"""
        return self.idbd.get_step_sizes()

    def update_step_sizes(self, gradients, features, td_error=None):
        """
        Update step-sizes for all parameters

        Args:
            gradients: list of parameter gradients
            features: input features (for eligibility)
            td_error: optional TD error

        Returns:
            list of step-sizes per parameter
        """
        # Flatten gradients
        flat_grad = np.concatenate([g.detach().cpu().numpy().flatten()
                                   for g in gradients])

        # Flatten features (use mean as proxy if needed)
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if features.ndim > 1:
            features = features.mean(axis=0)

        # Broadcast features to match gradient dimensions
        feature_vector = np.tile(features, len(flat_grad) // len(features) + 1)[:len(flat_grad)]

        # Update IDBD
        step_sizes = self.idbd.update(flat_grad, feature_vector, td_error)

        # Reshape step-sizes back to parameter shapes
        step_sizes_list = []
        start = 0
        for size, shape in zip(self.param_sizes, self.param_shapes):
            param_step_sizes = step_sizes[start:start+size].reshape(shape)
            step_sizes_list.append(torch.tensor(param_step_sizes))
            start += size

        return step_sizes_list

    def apply_updates(self, parameters, step_sizes_list):
        """Apply step-size-scaled updates to parameters"""
        for param, step_sizes in zip(parameters, step_sizes_list):
            if param.grad is not None:
                param.data -= step_sizes.to(param.device) * param.grad.data
