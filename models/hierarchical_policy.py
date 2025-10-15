"""
ARC hierarchical action selection utilities.

Provides a lightweight attention head that proposes structured actions
(operator + pointers) which are then evaluated by the existing Q-network.
This retains compatibility with the discrete Q-function while allowing
pointer-style selection for large action spaces.
"""
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class HierarchicalActionHead(nn.Module):
    """Attention-based proposer for ARC hierarchical actions."""

    SUPPORTED_OPERATORS = [
        'paint_cell',
        'fill_region',
        'mirror',
        'rotate',
        'submit',
    ]

    def __init__(self, state_encoder, env) -> None:
        super().__init__()
        self.encoder = state_encoder
        self.latent_dim = getattr(state_encoder, 'latent_dim', env.state_dim)
        self.grid_size = getattr(env, 'working_grid_size', 16)
        self.num_colors = getattr(env, 'NUM_COLORS', 10)
        self.max_examples = getattr(env, 'max_training_examples', 7)
        self.operator_names = list(getattr(env, 'operator_names', [])) or [
            'paint_cell',
            'fill_region',
            'copy_patch',
            'copy_exemplar_patch',
            'stamp_exemplar_output',
            'mirror',
            'rotate',
            'submit',
        ]
        self.num_operators = len(self.operator_names)

        self.operator_proj = nn.Linear(self.latent_dim, self.num_operators)
        self.grid_query = nn.Linear(self.latent_dim, self.latent_dim)
        self.color_proj = nn.Linear(self.latent_dim, self.num_colors)
        self.orientation_proj = nn.Linear(self.latent_dim, 4)

        for layer in (self.operator_proj, self.grid_query, self.color_proj, self.orientation_proj):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def _encode(self, state: np.ndarray) -> Tuple[torch.Tensor, dict]:
        device = next(self.operator_proj.parameters()).device
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            latent, tokens = self.encoder.encode_with_tokens(state_tensor)
        latent = latent.squeeze(0)
        grid_tokens = tokens['grid'].squeeze(0)
        return latent, {'grid': grid_tokens}

    def _pointer_scores(self, latent: torch.Tensor, grid_tokens: torch.Tensor) -> torch.Tensor:
        query = self.grid_query(latent)
        scores = torch.matmul(grid_tokens, query)
        return scores

    def _top_pointer(self, scores: torch.Tensor) -> int:
        if scores.numel() == 0:
            return 0
        return int(torch.argmax(scores).item())

    def generate_candidates(self, state: np.ndarray, num_candidates: int = 5) -> List[np.ndarray]:
        latent, tokens = self._encode(state)
        op_logits = self.operator_proj(latent)
        top_k = min(num_candidates, self.num_operators)
        top_ops = torch.topk(op_logits, k=top_k).indices.tolist()

        pointer_scores = self._pointer_scores(latent, tokens['grid'])
        pointer_idx = self._top_pointer(pointer_scores)
        color_idx = int(torch.argmax(self.color_proj(latent)).item())
        orientation_logits = self.orientation_proj(latent)

        candidates: List[np.ndarray] = []
        for op_idx in top_ops:
            op_name = self.operator_names[op_idx]
            if op_name not in self.SUPPORTED_OPERATORS:
                continue
            action_vec = np.full(6, -1, dtype=np.int64)
            action_vec[0] = op_idx
            if op_name in {'paint_cell', 'fill_region'}:
                action_vec[2] = pointer_idx
                action_vec[3] = color_idx
            elif op_name == 'mirror':
                axis_idx = int(torch.argmax(orientation_logits[:2]).item())
                action_vec[5] = axis_idx
            elif op_name == 'rotate':
                action_vec[5] = int(torch.argmax(orientation_logits).item())
            candidates.append(action_vec)

        # Always include submit as a fallback candidate
        submit_idx = self.operator_names.index('submit') if 'submit' in self.operator_names else None
        if submit_idx is not None:
            submit_vec = np.full(6, -1, dtype=np.int64)
            submit_vec[0] = submit_idx
            candidates.append(submit_vec)

        return candidates

    def sample_random(self, state: np.ndarray) -> np.ndarray:
        action_vec = np.full(6, -1, dtype=np.int64)
        random_op_name = np.random.choice(self.SUPPORTED_OPERATORS)
        op_idx = self.operator_names.index(random_op_name)
        action_vec[0] = op_idx
        if random_op_name in {'paint_cell', 'fill_region'}:
            action_vec[2] = np.random.randint(self.grid_size * self.grid_size)
            action_vec[3] = np.random.randint(self.num_colors)
        elif random_op_name == 'mirror':
            action_vec[5] = np.random.randint(2)
        elif random_op_name == 'rotate':
            action_vec[5] = np.random.randint(4)
        return action_vec


class ArcHierarchicalQWrapper:
    """Wrap a discrete Q-network with hierarchical action proposals for ARC."""

    def __init__(self, base_q, action_head: HierarchicalActionHead, env) -> None:
        self.base_q = base_q
        self.action_head = action_head
        self.env = env
        self.action_dim = getattr(base_q, 'action_dim', env.action_dim)

    def select_action(self, state, epsilon: float = 0.0):
        if np.random.rand() < epsilon:
            structured = self.action_head.sample_random(state)
            idx = self.env.action_index_from_structured(structured)
            if idx is not None:
                return idx
            return np.random.randint(self.action_dim)

        candidates = self.action_head.generate_candidates(state)
        scored_indices: List[Tuple[int, np.ndarray]] = []
        for structured in candidates:
            idx = self.env.action_index_from_structured(structured)
            if idx is not None:
                scored_indices.append((idx, structured))

        if not scored_indices:
            return self.base_q.select_action(state, epsilon=0.0)

        q_values = self.base_q.predict(state)
        best_idx = max(scored_indices, key=lambda item: q_values[item[0]])[0]
        return best_idx

    def __getattr__(self, name):  # Delegate other attributes/methods to base Q-network
        return getattr(self.base_q, name)

    def to(self, device):
        self.base_q = self.base_q.to(device)
        self.action_head = self.action_head.to(device)
        return self
