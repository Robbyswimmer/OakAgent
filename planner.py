"""
Planner with Dyna Imagination
Generates simulated experience using primitive and option models
Enables multi-timescale model-based planning (OaK principle #4 & #5)
"""
import numpy as np
import torch

class DynaPlanner:
    """
    Dyna planner with option-level abstractions
    Simulates rollouts mixing primitive actions and option jumps
    """

    def __init__(self, dyn_model, option_models, q_primitive, q_option,
                 option_library, config):
        self.dyn_model = dyn_model
        self.option_models = option_models
        self.q_primitive = q_primitive
        self.q_option = q_option
        self.option_library = option_library
        self.config = config

        # Planning parameters
        self.num_plan_steps = config.DYNA_PLAN_STEPS
        self.horizon = config.DYNA_HORIZON
        self.num_recent_states = config.DYNA_NUM_RECENT_STATES

        # Track planning statistics
        self.plan_count = 0
        self.primitive_sim_count = 0
        self.option_sim_count = 0

    def imagine_transitions(self, replay_buffer, num_simulations=None):
        """
        Generate simulated transitions using models

        Args:
            replay_buffer: buffer to sample start states from
            num_simulations: number of simulations (default: config value)

        Returns:
            list of (s, a, r, s', done) imagined transitions
        """
        if num_simulations is None:
            num_simulations = self.num_plan_steps

        if len(replay_buffer) == 0:
            return []

        simulated_transitions = []

        # Get recent states to start simulations from
        start_states = replay_buffer.get_recent_states(self.num_recent_states)
        if len(start_states) == 0:
            return []

        for _ in range(num_simulations):
            # Sample a random start state
            start_idx = np.random.randint(len(start_states))
            state = start_states[start_idx].copy()

            # Simulate a short rollout
            for step in range(self.horizon):
                # Decide: use primitive action or option?
                use_option = np.random.rand() < 0.3  # 30% option, 70% primitive

                if use_option and self.option_library.get_num_options() > 0:
                    # Simulate option execution
                    option_id = np.random.randint(self.option_library.get_num_options())
                    next_state, reward, duration = self.option_models.predict(option_id, state)

                    # Skip adding option transitions to primitive replay buffer
                    # (option Q-learning is done separately from executed option trajectories)
                    # Just update state for continued simulation
                    state = next_state
                    self.option_sim_count += 1

                else:
                    # Simulate primitive action
                    action = np.random.randint(self.q_primitive.action_dim)
                    next_state, reward = self.dyn_model.predict(state, action)

                    # Extract values from tensors
                    if isinstance(next_state, torch.Tensor):
                        next_state = next_state.squeeze().cpu().numpy()
                    if isinstance(reward, torch.Tensor):
                        reward = reward.squeeze().item()

                    simulated_transitions.append((
                        state.copy(),
                        action,
                        reward,
                        next_state,
                        False  # assume not terminal
                    ))

                    state = next_state
                    self.primitive_sim_count += 1

                # Early termination if state looks terminal
                if self._is_terminal_state(state):
                    break

        self.plan_count += 1

        return simulated_transitions

    def _is_terminal_state(self, state):
        """Check if state looks terminal (for CartPole)"""
        x = state[0]
        theta = state[2]

        # CartPole termination conditions
        x_threshold = 2.4
        theta_threshold = 0.2095  # ~12 degrees

        return (abs(x) > x_threshold or abs(theta) > theta_threshold)

    def plan_and_act(self, state, epsilon=0.0):
        """
        Use planning to select action or option

        Args:
            state: current state
            epsilon: exploration rate

        Returns:
            (action_or_option_id, is_option)
        """
        # Epsilon-greedy with mixing of primitives and options
        if np.random.rand() < epsilon:
            # Random exploration
            if np.random.rand() < 0.5 and self.option_library.get_num_options() > 0:
                # Random option
                option_id = np.random.randint(self.option_library.get_num_options())
                return option_id, True
            else:
                # Random primitive
                action = np.random.randint(self.q_primitive.action_dim)
                return action, False

        # Greedy selection based on Q-values
        q_primitive_values = self.q_primitive.predict(state)
        best_primitive_action = np.argmax(q_primitive_values)
        best_primitive_q = q_primitive_values[best_primitive_action]

        # Check option Q-values
        if self.option_library.get_num_options() > 0:
            q_option_values = self.q_option.predict(state)
            best_option_id = np.argmax(q_option_values)
            best_option_q = q_option_values[best_option_id]

            # Choose between best primitive and best option
            if best_option_q > best_primitive_q:
                return best_option_id, True

        return best_primitive_action, False

    def get_statistics(self):
        """Get planning statistics"""
        return {
            'plan_count': self.plan_count,
            'primitive_simulations': self.primitive_sim_count,
            'option_simulations': self.option_sim_count
        }


class MCPPlanner:
    """
    Model-Predictive Control (MPC) planner with options
    Alternative to Dyna: shooting-based planning
    """

    def __init__(self, dyn_model, option_models, config):
        self.dyn_model = dyn_model
        self.option_models = option_models
        self.config = config

        self.num_sequences = config.MPC_NUM_SEQUENCES
        self.horizon = config.MPC_HORIZON
        self.gamma = config.Q_GAMMA

    def plan(self, state, action_dim, num_options):
        """
        MPC planning: sample action sequences and evaluate

        Args:
            state: current state
            action_dim: number of primitive actions
            num_options: number of options

        Returns:
            best_action_or_option, is_option
        """
        best_value = float('-inf')
        best_action = 0
        best_is_option = False

        # Sample random action sequences
        for _ in range(self.num_sequences):
            # Generate random sequence of actions/options
            sequence = []
            for _ in range(self.horizon):
                # Mix primitives and options
                if np.random.rand() < 0.3 and num_options > 0:
                    # Option
                    sequence.append(('option', np.random.randint(num_options)))
                else:
                    # Primitive
                    sequence.append(('primitive', np.random.randint(action_dim)))

            # Evaluate sequence
            value = self._evaluate_sequence(state, sequence)

            # Track best
            if value > best_value:
                best_value = value
                first_action = sequence[0]
                if first_action[0] == 'option':
                    best_action = first_action[1]
                    best_is_option = True
                else:
                    best_action = first_action[1]
                    best_is_option = False

        return best_action, best_is_option

    def _evaluate_sequence(self, start_state, sequence):
        """Evaluate a sequence of actions/options using models"""
        state = start_state.copy()
        total_return = 0.0
        discount = 1.0

        for action_type, action_id in sequence:
            if action_type == 'option':
                # Use option model
                next_state, reward, duration = self.option_models.predict(action_id, state)
                total_return += discount * reward
                discount *= (self.gamma ** duration)
                state = next_state

            else:
                # Use primitive model
                next_state, reward = self.dyn_model.predict(state, action_id)
                if isinstance(next_state, torch.Tensor):
                    next_state = next_state.squeeze().cpu().numpy()
                if isinstance(reward, torch.Tensor):
                    reward = reward.squeeze().item()

                total_return += discount * reward
                discount *= self.gamma
                state = next_state

            # Check termination
            if self._is_terminal(state):
                break

        return total_return

    def _is_terminal(self, state):
        """Check if state is terminal"""
        x = state[0]
        theta = state[2]
        x_threshold = 2.4
        theta_threshold = 0.2095
        return (abs(x) > x_threshold or abs(theta) > theta_threshold)
