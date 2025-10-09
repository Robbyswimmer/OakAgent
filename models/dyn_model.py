"""
Primitive Dynamics Model
Ensemble of MLPs predicting Δs and r given (s, a)
Learned online for model-based planning (OaK principle #4)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta.meta_optimizer import MetaOptimizerAdapter

class DynamicsNet(nn.Module):
    """Single dynamics model: (s, a) -> (Δs, r)"""

    def __init__(self, state_dim, action_dim, hidden_size=128):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Input: state + action (one-hot)
        input_dim = state_dim + action_dim

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Delta state head (predicts Δs = s' - s)
        self.delta_head = nn.Linear(hidden_size, state_dim)

        # Reward head
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        """
        Args:
            state: (batch, state_dim) or (state_dim,)
            action: (batch,) int or (batch, action_dim) one-hot

        Returns:
            delta_state: (batch, state_dim)
            reward: (batch, 1)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Convert action to one-hot if needed
        # Handle various action shapes: (batch,), (batch, 1), or already one-hot (batch, action_dim)
        if action.dim() == 1:
            # Shape (batch,) - convert to one-hot (batch, action_dim)
            action_onehot = F.one_hot(action.long(), self.action_dim).float()
        elif action.dim() == 2 and action.shape[1] == 1:
            # Shape (batch, 1) - squeeze and convert to one-hot
            action_onehot = F.one_hot(action.squeeze(1).long(), self.action_dim).float()
        else:
            # Already one-hot (batch, action_dim)
            action_onehot = action.float()

        # Concatenate state and action
        x = torch.cat([state, action_onehot], dim=-1)

        # Forward pass
        features = self.shared(x)
        delta_state = self.delta_head(features)
        reward = self.reward_head(features)

        return delta_state, reward

    def predict(self, state, action):
        """Predict next state and reward"""
        delta_s, r = self.forward(state, action)
        next_state = state + delta_s
        return next_state, r


class DynamicsEnsemble:
    """
    Ensemble of dynamics models for uncertainty estimation
    Predicts p(s', r | s, a) via ensemble mean/variance
    """

    def __init__(self, state_dim, action_dim, ensemble_size=3, hidden_size=128, lr=1e-3, meta_config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size

        # Create ensemble
        self.models = [
            DynamicsNet(state_dim, action_dim, hidden_size)
            for _ in range(ensemble_size)
        ]

        meta_cfg = meta_config.copy() if meta_config is not None else None

        # Optimizers for each model
        self.optimizers = [
            MetaOptimizerAdapter(model.parameters(), base_lr=lr, meta_config=meta_cfg)
            for model in self.models
        ]

        # Track prediction errors
        self.prediction_errors = []

    def predict(self, state, action, return_std=False):
        """
        Predict next state and reward using ensemble

        Args:
            state: state tensor
            action: action tensor
            return_std: if True, also return standard deviation

        Returns:
            next_state_mean, reward_mean [, next_state_std, reward_std]
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, (int, np.integer)):
            action = torch.LongTensor([action])
        elif isinstance(action, np.ndarray):
            action = torch.LongTensor(action)

        predictions = []
        rewards = []

        for model in self.models:
            with torch.no_grad():
                next_state, reward = model.predict(state, action)
                predictions.append(next_state)
                rewards.append(reward)

        # Stack predictions
        predictions = torch.stack(predictions)  # (ensemble_size, batch, state_dim)
        rewards = torch.stack(rewards)  # (ensemble_size, batch, 1)

        # Compute mean and std
        next_state_mean = predictions.mean(dim=0)
        reward_mean = rewards.mean(dim=0)

        if return_std:
            next_state_std = predictions.std(dim=0)
            reward_std = rewards.std(dim=0)
            return next_state_mean, reward_mean, next_state_std, reward_std

        return next_state_mean, reward_mean

    def update(self, state, action, next_state, reward, lambda_r=1.0):
        """
        Update all models in ensemble with supervised learning

        Args:
            state: (batch, state_dim)
            action: (batch,) int
            next_state: (batch, state_dim)
            reward: (batch,)
            lambda_r: reward loss weight

        Returns:
            average loss across ensemble
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if isinstance(action, np.ndarray):
            action = torch.LongTensor(action)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state)
        if isinstance(reward, np.ndarray):
            reward = torch.FloatTensor(reward)

        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0) if action.dim() == 0 else action
            next_state = next_state.unsqueeze(0)
            reward = reward.unsqueeze(0) if reward.dim() == 0 else reward

        # Target delta
        target_delta = next_state - state
        target_reward = reward.unsqueeze(-1) if reward.dim() == 1 else reward

        total_loss = 0.0

        for model, optimizer in zip(self.models, self.optimizers):
            # Forward pass
            pred_delta, pred_reward = model(state, action)

            # Loss: MSE for delta_s and reward
            loss_delta = F.mse_loss(pred_delta, target_delta)
            loss_reward = F.mse_loss(pred_reward, target_reward)
            loss = loss_delta + lambda_r * loss_reward

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            feature_vec = state.detach().cpu().numpy().reshape(-1)
            optimizer.step(feature_vec, clip_range=(-10.0, 10.0))

            total_loss += loss.item()

        avg_loss = total_loss / self.ensemble_size

        # Track prediction error
        self.prediction_errors.append(avg_loss)

        return avg_loss

    def get_average_error(self, n=100):
        """Get average prediction error over last n updates"""
        if len(self.prediction_errors) == 0:
            return 0.0
        recent_errors = self.prediction_errors[-n:]
        return np.mean(recent_errors)

    def evaluate_errors(self, replay_buffer, batch_size=128, horizon=3):
        """Compute 1-step and multi-step MAE using the replay buffer."""
        if len(replay_buffer) == 0:
            return {'mae_1': 0.0, 'mae_3': 0.0, 'count_1': 0, 'count_3': 0}

        batch_size = min(batch_size, len(replay_buffer))
        states, actions, _, next_states, _ = replay_buffer.sample(batch_size)
        actions = actions.squeeze(-1)

        pred_next, _ = self.predict(states, actions)
        if isinstance(pred_next, torch.Tensor):
            pred_next = pred_next.cpu().numpy()

        mae_1 = float(np.mean(np.abs(pred_next - next_states)))

        sequences = replay_buffer.sample_sequences(horizon, batch_size)
        multi_errors = []
        for seq in sequences:
            s_pred = seq['states'][0]
            valid = True
            for step in range(horizon):
                action = int(seq['actions'][step])
                next_pred, _ = self.predict(s_pred, action)
                if isinstance(next_pred, torch.Tensor):
                    next_pred = next_pred.squeeze(0).cpu().numpy()
                if seq['dones'][step]:
                    valid = False
                    break
                s_pred = next_pred

            if not valid:
                continue

            target_state = seq['next_states'][horizon - 1]
            multi_errors.append(np.mean(np.abs(s_pred - target_state)))

        mae_3 = float(np.mean(multi_errors)) if multi_errors else 0.0
        return {
            'mae_1': mae_1,
            'mae_3': mae_3,
            'count_1': batch_size,
            'count_3': len(multi_errors),
        }
