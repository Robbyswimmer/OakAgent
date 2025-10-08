#!/bin/bash -l

#SBATCH --job-name="OaK-CartPole"
#SBATCH --output=logs/oak_cartpole_%j.txt
#SBATCH --error=logs/oak_cartpole_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=rmose009@ucr.edu

set -euo pipefail

echo "=========================================="
echo "OaK-CartPole Training Job"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started at: $(date)"
echo "=========================================="

# Configuration
PROJECT_ROOT=${PROJECT_ROOT:-"$PWD"}
VENV_PATH=${VENV_PATH:-"${PROJECT_ROOT}/venv"}
NUM_EPISODES=${NUM_EPISODES:-500}
SEED=${SEED:-42}
RUN_ABLATIONS=${RUN_ABLATIONS:-0}

# Create logs directory
mkdir -p logs
mkdir -p results

# Activate virtual environment
echo "Activating virtual environment at ${VENV_PATH}"
if [[ ! -d "$VENV_PATH" ]]; then
  echo "[ERROR] Virtual environment not found at ${VENV_PATH}" >&2
  echo "Please create venv first: python3 -m venv venv" >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"

# Verify dependencies
echo "Python version: $(python --version)"
echo "Installed packages:"
pip list | grep -E "gymnasium|torch|numpy"

# Run main training
echo ""
echo "=========================================="
echo "Starting OaK-CartPole Full Training"
echo "Episodes: ${NUM_EPISODES}"
echo "Seed: ${SEED}"
echo "=========================================="

python -u main.py \
  --num-episodes "${NUM_EPISODES}" \
  --seed "${SEED}"

echo ""
echo "Main training completed at $(date)"

# Run ablation studies if requested
if [[ "$RUN_ABLATIONS" != "0" ]]; then
  echo ""
  echo "=========================================="
  echo "Running Ablation Studies"
  echo "=========================================="

  ablations=("no_planning" "no_options" "no_gvfs" "no_idbd")

  for ablation in "${ablations[@]}"; do
    echo ""
    echo "--- Running ablation: ${ablation} ---"
    python -u main.py \
      --num-episodes "${NUM_EPISODES}" \
      --seed "${SEED}" \
      --ablation "${ablation}" \
      --output-suffix "_${ablation}"
    echo "--- Completed ablation: ${ablation} ---"
  done

  echo ""
  echo "All ablation studies completed at $(date)"
fi

# Summary
echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Completed at: $(date)"
echo "Results saved to: ${PROJECT_ROOT}/results/"
echo ""
echo "To view results:"
echo "  ls -lh results/"
echo "  cat logs/oak_cartpole_${SLURM_JOB_ID}.txt"
echo "=========================================="
