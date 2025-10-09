"""Utility wrappers for switching between standard optimizers and per-weight meta-learning."""
import numpy as np
import torch

from .idbd import TorchIDBD


class MetaOptimizerAdapter:
    """Wraps a parameter collection with either Adam or TorchIDBD-based updates."""

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

