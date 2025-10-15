"""Reward modeling utilities for ARC."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcRewardModel(nn.Module):
    """Predicts ARC transformation progress for potential-based shaping."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, lr: float = 1e-3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        return self.net(latent).squeeze(-1)

    def shaping(self, state_latent: torch.Tensor, next_latent: torch.Tensor) -> float:
        with torch.no_grad():
            current = self.forward(state_latent).mean()
            nxt = self.forward(next_latent).mean()
        return float((nxt - current).item())

    def update(self, latent: torch.Tensor, target: torch.Tensor) -> float:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        pred = self.forward(latent)
        loss = F.mse_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
