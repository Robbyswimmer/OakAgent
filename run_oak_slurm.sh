#!/bin/bash -l

#SBATCH --job-name="OaK-CartPole"
#SBATCH --output=logs/oak_training_%j.txt
#SBATCH --error=logs/oak_training_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu
#SBATCH -p gpu

# Exit on error, undefined variables, and pipe failures
set -euo pipefail

# Force unbuffered output for Python
export PYTHONUNBUFFERED=1

# ============================================================================
# OaK-CartPole SLURM Training Script
# Implements continual learning with FC-STOMP cycle
# ============================================================================

echo "=========================================="
echo "  OaK-CartPole Training on SLURM"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

# Conda environment activation
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v module &>/dev/null; then
  module load anaconda &>/dev/null || true
  source "$HOME/.bashrc"
fi

# Set conda environment name
CONDA_ENV=${CONDA_ENV:-oak-env}
echo "Activating conda environment '${CONDA_ENV}'"
conda activate "${CONDA_ENV}"

echo "✓ Environment activated"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# ============================================================================
# Configuration Variables (Override with environment variables)
# ============================================================================

# Environment selection
ENV_NAME=${ENV_NAME:-arc}  # Options: cartpole, arc

# Experiment paths
OUTPUT_ROOT=${OUTPUT_ROOT:-"$PWD/experiments/runs/${SLURM_JOB_ID}"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${OUTPUT_ROOT}/checkpoints"}
LOGS_DIR=${LOGS_DIR:-"$PWD/logs"}

# Training parameters
NUM_EPISODES=${NUM_EPISODES:-1000}
MAX_STEPS=${MAX_STEPS:-500}
EVAL_INTERVAL=${EVAL_INTERVAL:-50}
SEED=${SEED:-42}

# OaK-specific parameters
USE_CONTINUAL=${USE_CONTINUAL:-0}
INITIAL_REGIME=${INITIAL_REGIME:-R1_base}
FC_STOMP_INTERVAL=${FC_STOMP_INTERVAL:-500}
USE_DYNA=${USE_DYNA:-1}
DYNA_STEPS=${DYNA_STEPS:-20}

# Model parameters
BATCH_SIZE=${BATCH_SIZE:-256}
LR=${LR:-0.0003}
GAMMA=${GAMMA:-0.99}
EPSILON_START=${EPSILON_START:-1.0}
EPSILON_END=${EPSILON_END:-0.01}
EPSILON_DECAY=${EPSILON_DECAY:-0.995}

# IDBD/Meta-learning parameters
USE_IDBD=${USE_IDBD:-1}
IDBD_MU=${IDBD_MU:-0.01}

# Ablation mode (for ablation studies)
ABLATION=${ABLATION:-none}  # Options: none, no_planning, no_options, no_gvfs, no_idbd

# Logging
SAVE_FREQ=${SAVE_FREQ:-100}
DEBUG_LOGGING=${DEBUG_LOGGING:-0}

# ============================================================================
# ARC Environment Configuration
# ============================================================================

# ARC-specific parameters (only used if ENV_NAME=arc)
ARC_MAX_TRAINING_EXAMPLES=${ARC_MAX_TRAINING_EXAMPLES:-7}
ARC_REWARD_MODE=${ARC_REWARD_MODE:-binary}  # Options: binary, dense, shaped
ARC_WORKING_GRID_SIZE=${ARC_WORKING_GRID_SIZE:-16}
ARC_ACTION_STRIDE=${ARC_ACTION_STRIDE:-3}
ARC_MAX_PAINT_ACTIONS=${ARC_MAX_PAINT_ACTIONS:-1000}
ARC_MAX_FILL_ACTIONS=${ARC_MAX_FILL_ACTIONS:-500}
ARC_MAX_COPY_ACTIONS=${ARC_MAX_COPY_ACTIONS:-500}
ARC_MAX_EXEMPLAR_ACTIONS=${ARC_MAX_EXEMPLAR_ACTIONS:-1000}
ARC_DATA_PATH=${ARC_DATA_PATH:-data/arc}
ARC_REWARD_MODEL_LR=${ARC_REWARD_MODEL_LR:-0.001}

# ARC encoder parameters
ENCODER_LATENT_DIM=${ENCODER_LATENT_DIM:-256}
ENCODER_NUM_HEADS=${ENCODER_NUM_HEADS:-4}
ENCODER_NUM_LAYERS=${ENCODER_NUM_LAYERS:-2}

# ============================================================================
# Create Output Directories
# ============================================================================

mkdir -p "${LOGS_DIR}"
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${CHECKPOINT_DIR}"

echo "Output directory: ${OUTPUT_ROOT}"
echo "Checkpoint directory: ${CHECKPOINT_DIR}"
echo ""

# ============================================================================
# Save Configuration
# ============================================================================

CONFIG_FILE="${OUTPUT_ROOT}/training_config.json"
cat > "${CONFIG_FILE}" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "node": "$(hostname)",
  "start_time": "$(date +%Y-%m-%d_%H:%M:%S)",
  "conda_env": "${CONDA_ENV}",
  "env_name": "${ENV_NAME}",
  "num_episodes": ${NUM_EPISODES},
  "max_steps": ${MAX_STEPS},
  "seed": ${SEED},
  "use_continual": ${USE_CONTINUAL},
  "fc_stomp_interval": ${FC_STOMP_INTERVAL},
  "use_dyna": ${USE_DYNA},
  "dyna_steps": ${DYNA_STEPS},
  "batch_size": ${BATCH_SIZE},
  "learning_rate": ${LR},
  "gamma": ${GAMMA},
  "use_idbd": ${USE_IDBD},
  "idbd_mu": ${IDBD_MU},
  "ablation": "${ABLATION}",
  "arc_config": {
    "max_training_examples": ${ARC_MAX_TRAINING_EXAMPLES},
    "reward_mode": "${ARC_REWARD_MODE}",
    "working_grid_size": ${ARC_WORKING_GRID_SIZE},
    "action_stride": ${ARC_ACTION_STRIDE},
    "data_path": "${ARC_DATA_PATH}",
    "encoder_latent_dim": ${ENCODER_LATENT_DIM}
  }
}
EOF

echo "Configuration saved to ${CONFIG_FILE}"
echo ""

# ============================================================================
# Build Command Line Arguments (matching main.py argparse)
# ============================================================================

# Export environment variables for the config to use
export ARC_DATA_PATH="${ARC_DATA_PATH}"
export ARC_MAX_TRAINING_EXAMPLES="${ARC_MAX_TRAINING_EXAMPLES}"
export ARC_REWARD_MODE="${ARC_REWARD_MODE}"
export ARC_WORKING_GRID_SIZE="${ARC_WORKING_GRID_SIZE}"
export ARC_ACTION_STRIDE="${ARC_ACTION_STRIDE}"
export ARC_MAX_PAINT_ACTIONS="${ARC_MAX_PAINT_ACTIONS}"
export ARC_MAX_FILL_ACTIONS="${ARC_MAX_FILL_ACTIONS}"
export ARC_MAX_COPY_ACTIONS="${ARC_MAX_COPY_ACTIONS}"
export ARC_MAX_EXEMPLAR_ACTIONS="${ARC_MAX_EXEMPLAR_ACTIONS}"
export ARC_REWARD_MODEL_LR="${ARC_REWARD_MODEL_LR}"
export ENCODER_LATENT_DIM="${ENCODER_LATENT_DIM}"
export ENCODER_NUM_HEADS="${ENCODER_NUM_HEADS}"
export ENCODER_NUM_LAYERS="${ENCODER_NUM_LAYERS}"

# Determine config type
CONFIG_TYPE="default"
if [[ "${USE_CONTINUAL}" != "0" ]]; then
  CONFIG_TYPE="continual"
fi

# Build simple argument list (main.py only accepts these)
ARGS=(
  --env "${ENV_NAME}"
  --config-type "${CONFIG_TYPE}"
  --num-episodes "${NUM_EPISODES}"
  --seed "${SEED}"
)

# Add ablation if specified
if [[ "${ABLATION}" != "none" ]]; then
  ARGS+=(--ablation "${ABLATION}")
fi

# ============================================================================
# Run Training
# ============================================================================

echo "Starting OaK training with configuration:"
echo "  Environment: ${ENV_NAME}"
echo "  Episodes: ${NUM_EPISODES}"
echo "  Seed: ${SEED}"
echo "  FC-STOMP interval: ${FC_STOMP_INTERVAL}"
echo "  Dyna steps: ${DYNA_STEPS}"
echo "  IDBD enabled: ${USE_IDBD}"
echo "  Ablation mode: ${ABLATION}"

if [[ "${ENV_NAME,,}" == "arc" ]]; then
  echo ""
  echo "ARC Configuration:"
  echo "  Training examples: ${ARC_MAX_TRAINING_EXAMPLES}"
  echo "  Reward mode: ${ARC_REWARD_MODE}"
  echo "  Grid size: ${ARC_WORKING_GRID_SIZE}"
  echo "  Encoder latent dim: ${ENCODER_LATENT_DIM}"
fi

echo ""

# Run the main training script with unbuffered output
echo "[SHELL] About to invoke Python script..."
echo "[SHELL] Python version: $(python --version)"
echo "[SHELL] Python path: $(which python)"
echo "[SHELL] Starting training script NOW..."
python -u main.py "${ARGS[@]}"

TRAIN_EXIT_CODE=$?
echo "[SHELL] Python process exited with code: ${TRAIN_EXIT_CODE}"

echo ""
echo "=========================================="
if [[ ${TRAIN_EXIT_CODE} -eq 0 ]]; then
  echo "  Training Completed Successfully!"
else
  echo "  Training Failed (exit code: ${TRAIN_EXIT_CODE})"
fi
echo "=========================================="
echo ""

# ============================================================================
# Post-Training Analysis (Optional)
# ============================================================================

if [[ ${TRAIN_EXIT_CODE} -eq 0 ]] && [[ -f "visualize.py" ]]; then
  echo "Generating visualizations..."
  python visualize.py --output-dir "${OUTPUT_ROOT}" || true
  echo "✓ Visualizations saved to ${OUTPUT_ROOT}"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "Job Summary:"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Output directory: ${OUTPUT_ROOT}"
echo "  End time: $(date)"
echo "  Duration: $SECONDS seconds"
echo ""

# Save job metadata
cat > "${OUTPUT_ROOT}/job_metadata.json" <<EOF
{
  "job_id": "${SLURM_JOB_ID}",
  "node": "$(hostname)",
  "start_time": "$(date +%Y-%m-%d_%H:%M:%S)",
  "end_time": "$(date +%Y-%m-%d_%H:%M:%S)",
  "duration_seconds": ${SECONDS},
  "exit_code": ${TRAIN_EXIT_CODE},
  "output_dir": "${OUTPUT_ROOT}"
}
EOF

exit ${TRAIN_EXIT_CODE}
