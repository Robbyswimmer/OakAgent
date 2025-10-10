"""
OaK-CartPole Configuration
All hyperparameters following OaK principles from project_outline.md
"""
import math

class Config:
    # Environment
    ENV_NAME = "CartPole-v1"

    # Primitive Dynamics Model
    DYN_HIDDEN_SIZE = 128
    DYN_NUM_LAYERS = 2
    DYN_ENSEMBLE_SIZE = 3
    DYN_TRAIN_STEPS = 20  # M1: minibatches per env step
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
    GVF_GAMMA_LONG = 0.99  # for g4 (survival, normalized horizon)
    GVF_TRAIN_STEPS = 15  # M3
    GVF_BUFFER_SIZE = 1000  # ring buffer for feature mining
    GVF_LR = 1e-3
    GVF_META_ENABLED = True

    # Options
    OPTION_MAX_LENGTH = 10
    OPTION_THETA_THRESHOLD = 0.05  # radians
    OPTION_X_THRESHOLD = 0.1  # meters
    OPTION_VELOCITY_THRESHOLD = 0.5  # for stability
    OPTION_POLICY_HIDDEN = 64
    OPTION_MODEL_MIN_ROLLOUTS = 2
    OPTION_MODEL_ERROR_THRESHOLD = 2.0
    OPTION_PROTECTED_IDS = []  # Options shielded from pruning (default: none)
    OPTION_POLICY_LR = 1e-3
    OPTION_VALUE_LR = 1e-3
    OPTION_POLICY_META_ENABLED = True
    OPTION_VALUE_META_ENABLED = True
    OPTION_MODEL_LR = 1e-3
    OPTION_MODEL_META_ENABLED = True

    # Planner (Dyna)
    PLANNER_TYPE = "dyna"  # or "mpc"
    DYNA_PLAN_STEPS = 60  # M_plan
    DYNA_HORIZON = 11  # H
    DYNA_NUM_RECENT_STATES = 100  # B: recent states to sample from
    DYN_LR = 1e-3
    DYN_META_ENABLED = True

    # MPC (alternative planner)
    MPC_NUM_SEQUENCES = 100  # N
    MPC_HORIZON = 10

    # Meta-Learning (IDBD/TIDBD/Autostep)
    META_TYPE = "idbd"  # "idbd", "tidbd", or "autostep"
    META_MU = 3e-4  # meta learning rate
    META_INIT_LOG_ALPHA = math.log(1e-3)  # initial step-size
    META_MIN_ALPHA = 1e-6
    META_MAX_ALPHA = 0.1

    # FC-STOMP (post-normalization thresholds for GVF values ∈ [0,1])
    FC_STOMP_FREQ = 500  # T_FC: env steps between feature construction
    FC_FEATURE_VARIANCE_THRESHOLD = 0.04  # tighter gate once GVFs are normalized
    FC_FEATURE_INITIAL_VARIANCE = 0.06  # relaxed threshold early in training
    FC_FEATURE_RELAX_STEPS = 1200  # steps to use relaxed threshold
    FC_HISTORY_MIN_LENGTH = 20  # minimum samples before evaluating feature
    FC_MIN_CONTROLLABILITY = 0.10  # min action contrast via model lookahead (warm-up)
    FC_MODEL_CONTROLLABILITY_MIN = 0.20  # minimum model-based contrast to treat feature as controllable
    FC_MIN_CONTROLLABILITY_BOOTSTRAP = 0.05  # relaxed gate while no options exist
    FC_CONTROLLABILITY_H = 3  # horizon for model-based action contrast
    FC_FEATURE_SPAWN_COOLDOWN = 400  # per-feature cooldown between spawned options
    FC_OPTION_PRUNE_WINDOW = 500  # steps to evaluate option performance
    FC_OPTION_SUCCESS_THRESHOLD = 0.10  # min success rate to keep option
    FC_OPTION_PRUNE_RECENT_STARTS = 5  # min recent executions before windowed pruning
    FC_OPTION_PRUNE_MIN_AGE_STEPS = 1500  # minimum env steps before pruning new option
    FC_OPTION_MIN_EXECUTIONS = 20  # minimum executions before considering pruning
    FC_SURVIVAL_TARGET = 200.0
    FC_USAGE_WINDOW = 5  # episodes for option usage summaries

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
