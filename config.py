"""
OaK-CartPole Configuration
All hyperparameters following OaK principles from project_outline.md
"""
import numpy as np

class Config:
    # Environment
    ENV_NAME = "CartPole-v1"

    # Primitive Dynamics Model
    DYN_HIDDEN_SIZE = 128
    DYN_NUM_LAYERS = 2
    DYN_ENSEMBLE_SIZE = 3
    DYN_TRAIN_STEPS = 10  # M1: minibatches per env step
    DYN_BATCH_SIZE = 256
    DYN_LAMBDA_R = 1.0  # reward loss weight

    # Q-Networks (Primitive)
    Q_HIDDEN_SIZE = 128
    Q_NUM_LAYERS = 2
    Q_TARGET_SYNC_FREQ = 500  # steps
    Q_GAMMA = 0.99
    Q_TRAIN_STEPS = 10  # M2

    # Q-Networks (Options - SMDP)
    Q_OPTION_HIDDEN_SIZE = 128
    Q_OPTION_NUM_LAYERS = 2

    # GVFs (Knowledge Layer)
    GVF_HIDDEN_SIZE = 64
    GVF_GAMMA_SHORT = 0.97  # for g1, g2, g3
    GVF_GAMMA_LONG = 0.999  # for g4 (survival)
    GVF_TRAIN_STEPS = 10  # M3
    GVF_BUFFER_SIZE = 1000  # ring buffer for feature mining

    # Options
    OPTION_MAX_LENGTH = 10
    OPTION_THETA_THRESHOLD = 0.05  # radians
    OPTION_X_THRESHOLD = 0.1  # meters
    OPTION_VELOCITY_THRESHOLD = 0.5  # for stability
    OPTION_POLICY_HIDDEN = 64

    # Planner (Dyna)
    PLANNER_TYPE = "dyna"  # or "mpc"
    DYNA_PLAN_STEPS = 20  # M_plan
    DYNA_HORIZON = 5  # H
    DYNA_NUM_RECENT_STATES = 50  # B: recent states to sample from

    # MPC (alternative planner)
    MPC_NUM_SEQUENCES = 100  # N
    MPC_HORIZON = 10

    # Meta-Learning (IDBD/TIDBD/Autostep)
    META_TYPE = "idbd"  # "idbd", "tidbd", or "autostep"
    META_MU = 1e-3  # meta learning rate
    META_INIT_LOG_ALPHA = np.log(1e-3)  # initial step-size
    META_MIN_ALPHA = 1e-6
    META_MAX_ALPHA = 1.0

    # FC-STOMP
    FC_STOMP_FREQ = 1000  # T_FC: env steps between feature construction
    FC_FEATURE_VARIANCE_THRESHOLD = 0.1  # for stable features
    FC_MIN_CONTROLLABILITY = 0.3  # min correlation with actions
    FC_OPTION_PRUNE_WINDOW = 500  # steps to evaluate option performance
    FC_OPTION_SUCCESS_THRESHOLD = 0.2  # min success rate to keep option

    # Replay Buffers
    REPLAY_CAPACITY = 100000
    REPLAY_REAL_CAPACITY = 50000
    REPLAY_SIM_CAPACITY = 50000

    # Training
    NUM_EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 500
    EVAL_FREQ = 10  # episodes
    EVAL_EPISODES = 100
    TARGET_RETURN = 475.0  # CartPole solving threshold

    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY = 0.995

    # Logging
    LOG_FREQ = 1  # episodes
    LOG_DIR = "logs"
    SAVE_FREQ = 50  # episodes

    # Ablations (for evaluation)
    ABLATION_NO_PLANNING = False
    ABLATION_NO_OPTIONS = False
    ABLATION_NO_GVFS = False
    ABLATION_NO_IDBD = False

    @classmethod
    def get_config_dict(cls):
        """Return all config as dictionary"""
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith('_') and k.isupper()}
