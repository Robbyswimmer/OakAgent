import math
from dataclasses import dataclass

import pytest

from knowledge.feature_construct import FCSTOMPManager, FeatureConstructor


class DummyGVF:
    def __init__(self, name, history):
        self.name = name
        self._history = list(history)

    def get_recent_predictions(self, n=100):
        return self._history[-n:]

    def predict(self, state):
        # States in tests are simple floats so return directly.
        return float(state)


class DummyHorde:
    def __init__(self, histories):
        self._gvfs = {
            name: DummyGVF(name, history) for name, history in histories.items()
        }

    def get_prediction_histories(self):
        return {name: list(gvf.get_recent_predictions()) for name, gvf in self._gvfs.items()}

    def __getitem__(self, name):
        return self._gvfs[name]


class DummyOptionLibrary:
    def __init__(self):
        self.removed = []

    def get_statistics(self):
        return {
            0: {"executions": 12, "success_rate": 0.9},
            4: {"executions": 15, "success_rate": 0.1},
        }

    def remove_option(self, option_id):
        self.removed.append(option_id)


@dataclass
class DummyConfig:
    FC_FEATURE_VARIANCE_THRESHOLD: float = 0.2
    FC_MIN_CONTROLLABILITY: float = 0.3
    FC_OPTION_SUCCESS_THRESHOLD: float = 0.2
    FC_STOMP_FREQ: int = 5
    OPTION_THETA_THRESHOLD: float = 0.05
    OPTION_X_THRESHOLD: float = 0.1
    OPTION_VELOCITY_THRESHOLD: float = 0.5


@pytest.fixture
def fc_setup():
    histories = {
        "custom_feature": [0.25 + ((i % 5) * 0.005) for i in range(120)],
    }
    horde = DummyHorde(histories)
    option_library = DummyOptionLibrary()
    config = DummyConfig()
    manager = FCSTOMPManager(horde, option_library, option_models=None, q_option=None, config=config)
    return manager, option_library, config


def test_fc_stomp_cycle_logs_feature_and_prunes_option(fc_setup):
    manager, option_library, _ = fc_setup

    result = manager.run_fc_stomp_cycle(current_step=5, state_history=None, action_history=None)

    assert result["features_mined"] == 1
    assert result["subtasks_formed"] == 1
    assert result["options_pruned"] == 1
    assert option_library.removed == [4]
    assert manager.get_history()[-1] == result


def test_fc_stomp_frequency_control(fc_setup):
    manager, _, config = fc_setup

    assert not manager.should_run(0)
    assert not manager.should_run(config.FC_STOMP_FREQ - 1)
    assert manager.should_run(config.FC_STOMP_FREQ)

    manager.run_fc_stomp_cycle(current_step=config.FC_STOMP_FREQ, state_history=None, action_history=None)

    assert not manager.should_run(config.FC_STOMP_FREQ * 2 - 1)
    assert manager.should_run(config.FC_STOMP_FREQ * 2)


def test_controllability_analysis_uses_correlation(fc_setup):
    manager, _, _ = fc_setup
    constructor: FeatureConstructor = manager.feature_constructor

    states = [0.0, 1.0, 0.0, 1.0] * 15
    actions = [0, 1, 0, 1] * 15

    correlation = constructor.analyze_feature_controllability("custom_feature", states, actions)

    assert math.isclose(correlation, 1.0, rel_tol=1e-5)
