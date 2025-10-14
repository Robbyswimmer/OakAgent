"""
Replay Buffers for OaK Agent
- RB_real: stores real environment transitions
- RB_sim: stores simulated (Dyna) transitions
- Trajectory storage for option rollouts
"""
import numpy as np
from collections import deque

class ReplayBuffer:
    """Ring buffer for storing transitions"""

    def __init__(self, capacity, state_dim, action_dim=1):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Pre-allocate arrays for efficiency (main buffer)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self.ptr = 0
        self.size = 0

        # Reservoir buffer for continual learning (Fix #5) - preserve old regime data
        self.reservoir_size = int(capacity * 0.1)  # 10% for long-term memory
        self.reservoir_states = np.zeros((self.reservoir_size, state_dim), dtype=np.float32)
        self.reservoir_actions = np.zeros((self.reservoir_size, action_dim), dtype=np.int64)
        self.reservoir_rewards = np.zeros(self.reservoir_size, dtype=np.float32)
        self.reservoir_next_states = np.zeros((self.reservoir_size, state_dim), dtype=np.float32)
        self.reservoir_dones = np.zeros(self.reservoir_size, dtype=np.bool_)
        self.reservoir_ptr = 0
        self.reservoir_count = 0

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        # Add to main buffer
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # Reservoir sampling (Fix #5): Probabilistically add to reservoir for long-term memory
        # 10% chance to add to reservoir (if not full, always add; if full, replace randomly)
        if np.random.rand() < 0.1:
            if self.reservoir_count < self.reservoir_size:
                # Reservoir not full, add to next slot
                self.reservoir_states[self.reservoir_ptr] = state
                self.reservoir_actions[self.reservoir_ptr] = action
                self.reservoir_rewards[self.reservoir_ptr] = reward
                self.reservoir_next_states[self.reservoir_ptr] = next_state
                self.reservoir_dones[self.reservoir_ptr] = done
                self.reservoir_ptr = (self.reservoir_ptr + 1) % self.reservoir_size
                self.reservoir_count += 1
            else:
                # Reservoir full, replace random entry (classic reservoir sampling)
                replace_idx = np.random.randint(0, self.reservoir_size)
                self.reservoir_states[replace_idx] = state
                self.reservoir_actions[replace_idx] = action
                self.reservoir_rewards[replace_idx] = reward
                self.reservoir_next_states[replace_idx] = next_state
                self.reservoir_dones[replace_idx] = done

    def sample(self, batch_size, mix_reservoir=True):
        """
        Sample a random batch of transitions (Fix #5: optionally mix with reservoir)

        Args:
            batch_size: Number of transitions to sample
            mix_reservoir: If True, mix 20% reservoir samples with 80% main buffer

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if not mix_reservoir or self.reservoir_count == 0:
            # Original sampling (no reservoir mixing)
            indices = np.random.randint(0, self.size, size=batch_size)
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_states[indices],
                self.dones[indices]
            )

        # Mix reservoir (20%) with main buffer (80%)
        reservoir_samples = max(1, int(batch_size * 0.2))
        main_samples = batch_size - reservoir_samples

        # Sample from both buffers
        main_indices = np.random.randint(0, self.size, size=main_samples)
        reservoir_indices = np.random.randint(0, self.reservoir_count, size=reservoir_samples)

        # Combine samples
        states = np.concatenate([
            self.states[main_indices],
            self.reservoir_states[reservoir_indices]
        ])
        actions = np.concatenate([
            self.actions[main_indices],
            self.reservoir_actions[reservoir_indices]
        ])
        rewards = np.concatenate([
            self.rewards[main_indices],
            self.reservoir_rewards[reservoir_indices]
        ])
        next_states = np.concatenate([
            self.next_states[main_indices],
            self.reservoir_next_states[reservoir_indices]
        ])
        dones = np.concatenate([
            self.dones[main_indices],
            self.reservoir_dones[reservoir_indices]
        ])

        return (states, actions, rewards, next_states, dones)

    def _chronological_index(self, idx):
        """Map chronological index to buffer slot."""
        return (self.ptr - self.size + idx) % self.capacity

    def sample_sequences(self, length, batch_size):
        """Sample contiguous sequences of transitions of given length."""
        if self.size < length or length <= 0:
            return []

        sequences = []
        max_start = self.size - length
        attempts = 0
        max_attempts = batch_size * 5

        while len(sequences) < batch_size and attempts < max_attempts:
            start = np.random.randint(0, max_start + 1)
            indices = [self._chronological_index(start + offset) for offset in range(length)]

            states_seq = self.states[indices]
            actions_seq = self.actions[indices].squeeze(-1)
            rewards_seq = self.rewards[indices]
            next_states_seq = self.next_states[indices]
            dones_seq = self.dones[indices]

            # Require trajectory to stay non-terminal until final transition
            if np.any(dones_seq[:-1]):
                attempts += 1
                continue

            sequences.append({
                'states': states_seq.copy(),
                'actions': actions_seq.copy(),
                'rewards': rewards_seq.copy(),
                'next_states': next_states_seq.copy(),
                'dones': dones_seq.copy(),
            })
            attempts += 1

        return sequences

    def get_recent_states(self, n):
        """Get n most recent states for Dyna planning"""
        if self.size == 0:
            return np.array([])

        n = min(n, self.size)
        if self.ptr >= n:
            indices = np.arange(self.ptr - n, self.ptr)
        else:
            # Wrap around
            indices = np.concatenate([
                np.arange(self.capacity - (n - self.ptr), self.capacity),
                np.arange(0, self.ptr)
            ])
        return self.states[indices]

    def __len__(self):
        return self.size


class TrajectoryBuffer:
    """
    Buffer for storing option execution trajectories
    Each trajectory is a sequence of (s, a, r, s', done) tuples
    """

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.trajectories = deque(maxlen=capacity)

    def add_trajectory(self, trajectory):
        """
        Add a complete trajectory
        trajectory: list of (s, a, r, s', done) tuples
        """
        self.trajectories.append(trajectory)

    def sample_trajectory(self):
        """Sample a random trajectory"""
        if len(self.trajectories) == 0:
            return []
        idx = np.random.randint(len(self.trajectories))
        return self.trajectories[idx]

    def get_all_trajectories(self):
        """Get all stored trajectories"""
        return list(self.trajectories)

    def __len__(self):
        return len(self.trajectories)


class OptionTrajectory:
    """
    Stores a single option execution trajectory
    Computes SMDP quantities: cumulative reward R_o, duration τ, start/end states
    """

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def add_step(self, state, action, reward, done):
        """Add a step to the trajectory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def get_smdp_data(self):
        """
        Compute SMDP quantities for option model learning
        Returns: (s_start, s_end, R_total, duration, trajectory)
        """
        if len(self.states) == 0:
            return None

        s_start = self.states[0]
        s_end = self.states[-1]
        R_total = sum(self.rewards)
        duration = len(self.rewards)

        trajectory = list(zip(
            self.states[:-1],
            self.actions,
            self.rewards,
            self.states[1:] if len(self.states) > 1 else [self.states[0]],
            self.dones
        ))

        return s_start, s_end, R_total, duration, trajectory

    def __len__(self):
        return len(self.states)
