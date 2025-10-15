"""
State Encoder modules for OaK agents.

Provides a registry of encoders that transform raw environment states into
latent representations shared across models. The ARC encoder uses a CNN +
Transformer stack over grid tokens and exemplar summaries, while the default
identity encoder simply passes vectors through unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    """Lightweight configuration for building state encoders."""

    input_dim: int
    latent_dim: int
    working_grid_size: Optional[int] = None
    num_colors: Optional[int] = None
    max_training_examples: Optional[int] = None
    exemplar_feature_dim: Optional[int] = None
    num_attention_heads: int = 4
    transformer_layers: int = 2


class StateEncoderBase(nn.Module):
    """Base class for state encoders."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim

    def encode_tensor(self, state: torch.Tensor) -> torch.Tensor:
        """Encode a batch of states (tensor) into latent embeddings."""
        raise NotImplementedError

    def encode_numpy(self, state: np.ndarray) -> np.ndarray:
        """Encode a numpy state array into latent numpy array."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            latent = self.encode_tensor(state_tensor)
        latent_np = latent.detach().cpu().numpy()
        if state.ndim == 1:
            return latent_np[0]
        return latent_np

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encode_tensor(state)


class IdentityEncoder(StateEncoderBase):
    """Pass-through encoder for vector observations."""

    def __init__(self, input_dim: int) -> None:
        super().__init__(EncoderConfig(input_dim=input_dim, latent_dim=input_dim))

    def encode_tensor(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            return state.unsqueeze(0)
        return state


class ArcVisualContextEncoder(StateEncoderBase):
    """CNN + Transformer encoder for ARC grid tasks."""

    def __init__(self, config: EncoderConfig) -> None:
        if config.working_grid_size is None or config.max_training_examples is None:
            raise ValueError("ARC encoder requires grid and exemplar configuration")
        super().__init__(config)

        grid_channels = 1
        latent_channels = max(config.latent_dim // 4, 32)

        self.grid_size = config.working_grid_size
        self.grid_tokens_side = self.grid_size
        self.num_colors = config.num_colors or 10
        self.max_examples = config.max_training_examples
        token_dim = config.exemplar_feature_dim or (2 * self.num_colors + 3)
        self.token_dim = token_dim
        self.context_dim = 8
        spatial_dim = config.input_dim - (self.grid_size * self.grid_size) - (self.max_examples * token_dim) - self.context_dim
        self.spatial_feature_dim = max(0, spatial_dim)

        # CNN backbone for spatial grid tokens
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(grid_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, latent_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_channels, config.latent_dim, kernel_size=1),
        )
        self.grid_pool = nn.AdaptiveAvgPool2d((self.grid_tokens_side, self.grid_tokens_side))

        # Projections for exemplar tokens and global context
        self.exemplar_proj = nn.Linear(token_dim, config.latent_dim)
        self.context_proj = nn.Linear(self.context_dim, config.latent_dim)
        if self.spatial_feature_dim > 0:
            self.spatial_mlp = nn.Sequential(
                nn.Linear(self.spatial_feature_dim, config.latent_dim),
                nn.ReLU(),
                nn.Linear(config.latent_dim, config.latent_dim),
            )
        else:
            self.spatial_mlp = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.latent_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)

        total_tokens = 1 + self.max_examples + self.grid_tokens_side * self.grid_tokens_side
        self.positional = nn.Parameter(torch.zeros(1, total_tokens, config.latent_dim))
        nn.init.trunc_normal_(self.positional, std=0.02)

        # Fusion MLP to aggregate attended tokens
        fusion_input_dim = config.latent_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )

    def _split_components(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split flattened state into grid, exemplar, and context parts."""
        grid_flat_dim = self.grid_size * self.grid_size
        exemplar_dim = self.max_examples * self.token_dim
        spatial_dim = self.spatial_feature_dim
        grid_part = state[:, :grid_flat_dim]
        spatial_part = state[:, grid_flat_dim : grid_flat_dim + spatial_dim]
        exemplar_part = state[:, grid_flat_dim + spatial_dim : grid_flat_dim + spatial_dim + exemplar_dim]
        context_part = state[:, grid_flat_dim + spatial_dim + exemplar_dim : grid_flat_dim + spatial_dim + exemplar_dim + self.context_dim]
        return grid_part, spatial_part, exemplar_part, context_part

    def _encode_components(self, state: torch.Tensor):
        grid_part, spatial_part, exemplar_part, context_part = self._split_components(state)

        grid_map = grid_part.view(-1, 1, self.grid_size, self.grid_size)
        grid_features = self.grid_encoder(grid_map)
        grid_features = self.grid_pool(grid_features)
        grid_tokens = grid_features.permute(0, 2, 3, 1).reshape(state.size(0), -1, self.latent_dim)

        spatial_tokens = torch.zeros(state.size(0), 0, self.latent_dim, device=state.device)

        exemplar_tokens = exemplar_part.view(-1, self.max_examples, self.token_dim)
        exemplar_tokens = self.exemplar_proj(exemplar_tokens)

        context_token = self.context_proj(context_part).unsqueeze(1)

        tokens = torch.cat([context_token, exemplar_tokens, spatial_tokens, grid_tokens], dim=1)
        pos = self.positional[:, : tokens.size(1)]
        tokens = tokens + pos

        attended = self.transformer(tokens)

        context_updated = attended[:, 0]
        exemplar_updated = attended[:, 1 : 1 + self.max_examples]
        spatial_offset = 1 + self.max_examples
        spatial_updated = attended[:, spatial_offset : spatial_offset + spatial_tokens.size(1)]
        grid_updated = attended[:, spatial_offset + spatial_tokens.size(1) :]

        exemplar_summary = exemplar_updated.mean(dim=1)
        if self.spatial_mlp is not None and spatial_part.numel() > 0:
            spatial_summary = self.spatial_mlp(spatial_part)
        else:
            spatial_summary = torch.zeros_like(exemplar_summary)
        grid_summary = grid_updated.mean(dim=1)

        fused = torch.cat([context_updated, exemplar_summary, spatial_summary, grid_summary], dim=1)
        latent = self.fusion(fused)

        token_dict = {
            'context': context_updated,
            'exemplar': exemplar_updated,
            'spatial': spatial_updated if spatial_tokens.size(1) > 0 else torch.zeros_like(exemplar_updated[:, :0, :]),
            'grid': grid_updated,
        }
        return latent, token_dict

    def encode_tensor(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.float()

        latent, _ = self._encode_components(state)
        return latent

    def encode_with_tokens(self, state: torch.Tensor):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.float()
        latent, token_dict = self._encode_components(state)
        return latent, token_dict


def create_state_encoder(env_name: str, env, config) -> StateEncoderBase:
    """Factory function that builds the appropriate encoder for an environment."""
    env_key = env_name.lower()
    if env_key in {"cartpole", "cartpole-v1", "cartpole_continual", "continual_cartpole"}:
        encoder = IdentityEncoder(input_dim=env.state_dim)
        encoder.eval()
        return encoder

    if env_key in {"arc"}:
        working_size = getattr(env, "working_grid_size", getattr(config, "ARC_WORKING_GRID_SIZE", 16))
        max_examples = getattr(env, "max_training_examples", getattr(config, "ARC_MAX_TRAINING_EXAMPLES", 7))
        latent_dim = getattr(config, "ENCODER_LATENT_DIM", 256)
        num_heads = getattr(config, "ENCODER_NUM_HEADS", 4)
        transformer_layers = getattr(config, "ENCODER_NUM_LAYERS", 2)
        num_colors = getattr(env, "NUM_COLORS", getattr(config, "NUM_COLORS", 10))
        token_dim = 2 * num_colors + 3
        encoder_config = EncoderConfig(
            input_dim=env.state_dim,
            latent_dim=latent_dim,
            working_grid_size=working_size,
            num_colors=num_colors,
            max_training_examples=max_examples,
            exemplar_feature_dim=token_dim,
            num_attention_heads=num_heads,
            transformer_layers=transformer_layers,
        )
        encoder = ArcVisualContextEncoder(encoder_config)
        encoder.eval()
        return encoder

    raise ValueError(f"No state encoder registered for environment '{env_name}'")


__all__ = [
    "EncoderConfig",
    "StateEncoderBase",
    "IdentityEncoder",
    "ArcVisualContextEncoder",
    "create_state_encoder",
]
