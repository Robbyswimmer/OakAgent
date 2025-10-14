"""
CartPole Environment Package for OaK Framework
Includes standard and continual learning variants
"""
from .env import CartPoleEnv
from .continual_env import ContinualCartPoleEnv, create_regime_schedule
from .gvfs import CartPoleHordeGVFs
from .options import UprightOption, CenteringOption, StabilizeOption, BalanceOption

__all__ = [
    'CartPoleEnv',
    'ContinualCartPoleEnv',
    'create_regime_schedule',
    'CartPoleHordeGVFs',
    'UprightOption',
    'CenteringOption',
    'StabilizeOption',
    'BalanceOption',
]
