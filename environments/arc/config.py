"""
ARC Configuration
All hyperparameters for ARC environment following OaK principles
"""
import math


class ARCConfig:
    # Environment
    ENV_NAME = "ARC"
    MAX_GRID_SIZE = 30  # ARC grids up to 30×30
    NUM_COLORS = 10     # Colors 0-9
    MAX_EPISODE_STEPS = 50  # Steps per episode

    # Primitive Dynamics Model
    DYN_HIDDEN_SIZE = 256  # Larger for complex transformations
    DYN_NUM_LAYERS = 3
    DYN_ENSEMBLE_SIZE = 5  # More ensemble members for uncertainty
    DYN_TRAIN_STEPS = 30  # M1: minibatches per env step
    DYN_BATCH_SIZE = 128
    DYN_LAMBDA_R = 1.0
    DYN_LR = 1e-3
    DYN_META_ENABLED = True

    # Q-Networks (Primitive)
    Q_HIDDEN_SIZE = 256
    Q_NUM_LAYERS = 3
    Q_TARGET_SYNC_FREQ = 1000  # steps
    Q_GAMMA = 0.95  # Lower gamma for shorter horizons
    Q_TRAIN_STEPS = 20  # M2

    # Q-Networks (Options - SMDP)
    Q_OPTION_HIDDEN_SIZE = 256
    Q_OPTION_NUM_LAYERS = 3

    # GVFs (Knowledge Layer)
    GVF_HIDDEN_SIZE = 128
    GVF_GAMMA = 0.95  # Short-term predictions for grid properties
    GVF_TRAIN_STEPS = 25  # M3
    GVF_BUFFER_SIZE = 500  # Ring buffer for feature mining
    GVF_LR = 3e-3
    GVF_META_ENABLED = True

    # Options
    OPTION_MAX_LENGTH = 15  # Longer for complex transformations
    OPTION_POLICY_HIDDEN = 128
    OPTION_MODEL_MIN_ROLLOUTS = 3
    OPTION_MODEL_ERROR_THRESHOLD = 1.5
    OPTION_PROTECTED_IDS = []
    OPTION_POLICY_LR = 1e-3
    OPTION_VALUE_LR = 1e-3
    OPTION_POLICY_META_ENABLED = True
    OPTION_VALUE_META_ENABLED = True
    OPTION_MODEL_LR = 1e-3
    OPTION_MODEL_META_ENABLED = True

    # Planner (Dyna)
    PLANNER_TYPE = "dyna"
    DYNA_PLAN_STEPS = 100  # M_plan (more planning for sparse rewards)
    DYNA_HORIZON = 20  # H (longer horizons for complex tasks)
    DYNA_NUM_RECENT_STATES = 200  # B: recent states to sample from
    OAK_PURITY_MODE = True  # Streaming + on-demand model sampling

    # MPC (alternative planner)
    MPC_NUM_SEQUENCES = 150  # N
    MPC_HORIZON = 15

    # Meta-Learning (IDBD/TIDBD/Autostep)
    META_TYPE = "idbd"
    META_MU = 5e-4  # meta learning rate (higher for faster adaptation)
    META_INIT_LOG_ALPHA = math.log(1e-3)
    META_MIN_ALPHA = 1e-6
    META_MAX_ALPHA = 0.1

    # FC-STOMP (post-normalization thresholds for GVF values)
    FC_STOMP_FREQ = 100  # T_FC: env steps between feature construction
    FC_FEATURE_VARIANCE_THRESHOLD = 0.03
    FC_FEATURE_INITIAL_VARIANCE = 0.05
    FC_FEATURE_RELAX_STEPS = 500
    FC_HISTORY_MIN_LENGTH = 30
    FC_MIN_CONTROLLABILITY = 0.10
    FC_MODEL_CONTROLLABILITY_MIN = 0.15
    FC_MIN_CONTROLLABILITY_BOOTSTRAP = 0.05
    FC_CONTROLLABILITY_H = 5
    FC_FEATURE_SPAWN_COOLDOWN = 100
    FC_OPTION_PRUNE_WINDOW = 500
    FC_OPTION_SUCCESS_THRESHOLD = 0.10
    FC_OPTION_PRUNE_RECENT_STARTS = 5
    FC_OPTION_PRUNE_MIN_AGE_STEPS = 1000
    FC_OPTION_MIN_EXECUTIONS = 20

    # Experience Buffers
    REPLAY_CAPACITY = 50000
    REPLAY_REAL_CAPACITY = 25000  # Real experience buffer
    REPLAY_SIM_CAPACITY = 25000   # Simulated experience buffer

    # Training
    NUM_EPISODES = 500  # Fewer episodes (ARC tasks are harder)
    MAX_STEPS_PER_EPISODE = 50
    EVAL_FREQ = 25  # episodes
    EVAL_EPISODES = 50
    EVAL_EPISODES_FINAL = 100
    TARGET_RETURN = 0.5  # ARC solving threshold (50% success rate)

    # Exploration
    EPSILON_START = 1.0
    EPSILON_END = 0.05  # Higher floor for harder exploration
    EPSILON_DECAY = 0.99

    # Logging
    LOG_FREQ = 5  # episodes
    LOG_DIR = "logs_arc"
    SAVE_FREQ = 25  # episodes

    # Ablations
    ABLATION_NO_PLANNING = False
    ABLATION_NO_OPTIONS = False
    ABLATION_NO_GVFS = False
    ABLATION_NO_IDBD = False

    # ARC-Specific
    ARC_DATA_PATH = "data/arc"  # Path to ARC task files
    ARC_TRAIN_TASKS = None  # List of training task IDs (None = use all)
    ARC_TEST_TASKS = None   # List of test task IDs

    @classmethod
    def get_config_dict(cls):
        """Return all config as dictionary"""
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith('_') and k.isupper()}
