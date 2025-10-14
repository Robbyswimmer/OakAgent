"""
Environment Factory for OaK Framework
Creates environment-specific components (env, GVFs, config)
"""
from typing import Tuple, Any


def create_environment(env_name: str, **kwargs):
    """
    Create an OaK environment by name.

    Args:
        env_name: Environment identifier ('cartpole', 'cartpole_continual', 'arc')
        **kwargs: Environment-specific arguments

    Returns:
        environment: OaKEnvironment instance

    Example:
        env = create_environment('cartpole')
        env = create_environment('cartpole_continual', regime='R1_base')
        env = create_environment('arc', task_path='path/to/task.json')
    """
    if env_name in ['cartpole', 'CartPole', 'cartpole-v1']:
        from environments.cartpole import CartPoleEnv
        return CartPoleEnv()

    elif env_name in ['cartpole_continual', 'continual_cartpole', 'cartpole-continual']:
        from environments.cartpole import ContinualCartPoleEnv
        regime = kwargs.get('regime', 'R1_base')
        return ContinualCartPoleEnv(regime=regime)

    elif env_name in ['arc', 'ARC']:
        from environments.arc import ARCEnvironment
        return ARCEnvironment(**kwargs)

    else:
        raise ValueError(
            f"Unknown environment: {env_name}. "
            f"Supported: cartpole, cartpole_continual, arc"
        )


def create_gvf_horde(env_name: str, state_dim: int, config: Any, meta_config: Any = None):
    """
    Create environment-specific GVF Horde.

    Args:
        env_name: Environment identifier
        state_dim: State dimension
        config: Configuration object
        meta_config: Meta-learning configuration (optional)

    Returns:
        horde: Environment-specific Horde instance

    Example:
        horde = create_gvf_horde('cartpole', state_dim=4, config=config)
    """
    if env_name in ['cartpole', 'CartPole', 'cartpole-v1', 'cartpole_continual', 'continual_cartpole']:
        from environments.cartpole.gvfs import CartPoleHordeGVFs
        return CartPoleHordeGVFs(state_dim, config, meta_config)

    elif env_name in ['arc', 'ARC']:
        from environments.arc.gvfs import ARCHordeGVFs
        return ARCHordeGVFs(state_dim, config, meta_config)

    else:
        raise ValueError(
            f"Unknown environment for GVF creation: {env_name}. "
            f"Supported: cartpole, cartpole_continual, arc"
        )


def load_config(env_name: str, config_type: str = 'default'):
    """
    Load environment-specific configuration.

    Args:
        env_name: Environment identifier
        config_type: Configuration type ('default', 'continual', etc.)

    Returns:
        config: Configuration class

    Example:
        Config = load_config('cartpole', 'default')
        config = Config()
    """
    if env_name in ['cartpole', 'CartPole', 'cartpole-v1']:
        if config_type == 'continual':
            from environments.cartpole.continual_config import ContinualConfig
            return ContinualConfig
        else:
            from environments.cartpole.config import Config
            return Config

    elif env_name in ['arc', 'ARC']:
        from environments.arc.config import ARCConfig
        return ARCConfig

    else:
        raise ValueError(
            f"Unknown environment for config loading: {env_name}. "
            f"Supported: cartpole, arc"
        )


def get_environment_info(env_name: str) -> dict:
    """
    Get metadata about an environment.

    Args:
        env_name: Environment identifier

    Returns:
        info: Dictionary with environment metadata
            - full_name: Full environment name
            - description: Brief description
            - state_type: 'vector' or 'grid'
            - action_type: 'discrete' or 'continuous'
            - has_continual_variant: Whether continual learning variant exists
    """
    info = {
        'cartpole': {
            'full_name': 'CartPole-v1',
            'description': 'Classic cart-pole balancing task',
            'state_type': 'vector',
            'action_type': 'discrete',
            'has_continual_variant': True,
        },
        'arc': {
            'full_name': 'Abstract Reasoning Challenge (ARC)',
            'description': 'Grid-based abstract reasoning puzzles',
            'state_type': 'grid',
            'action_type': 'discrete',
            'has_continual_variant': False,
        }
    }

    # Normalize environment name
    env_key = env_name.lower()
    if env_key in ['cartpole-v1', 'cartpole_continual', 'continual_cartpole']:
        env_key = 'cartpole'

    if env_key not in info:
        raise ValueError(f"Unknown environment: {env_name}")

    return info[env_key]
