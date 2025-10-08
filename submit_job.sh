#!/bin/bash
# Helper script to submit SLURM job with different configurations

print_usage() {
  cat <<EOF
Usage: ./submit_job.sh [OPTIONS]

Options:
  --episodes N        Number of training episodes (default: 500)
  --ablations         Run ablation studies after main training
  --time HH:MM:SS     SLURM time limit (default: 24:00:00)
  --mem N             Memory in GB (default: 16)
  --cpus N            CPUs per task (default: 4)
  --partition NAME    SLURM partition to use
  -h, --help          Show this help message

Examples:
  # Basic training run
  ./submit_job.sh --episodes 500

  # Full training with ablations
  ./submit_job.sh --episodes 1000 --ablations

  # Short test run
  ./submit_job.sh --episodes 100 --time 2:00:00 --mem 8

EOF
}

# Default values
NUM_EPISODES=500
RUN_ABLATIONS=0
TIME_LIMIT="24:00:00"
MEMORY="16G"
CPUS=4
PARTITION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --episodes)
      NUM_EPISODES="$2"
      shift 2
      ;;
    --ablations)
      RUN_ABLATIONS=1
      shift
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --mem)
      MEMORY="${2}G"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      print_usage
      exit 1
      ;;
  esac
done

# Set environment variables for SLURM script
export NUM_EPISODES
export RUN_ABLATIONS
export PROJECT_ROOT="$PWD"

# Build sbatch command
SBATCH_CMD="sbatch"

# Override SLURM directives if specified
OVERRIDE_ARGS=""
if [[ -n "$PARTITION" ]]; then
  OVERRIDE_ARGS="${OVERRIDE_ARGS} --partition=${PARTITION}"
fi

# Submit job
echo "Submitting OaK-CartPole training job..."
echo "  Episodes: ${NUM_EPISODES}"
echo "  Ablations: ${RUN_ABLATIONS}"
echo "  Time limit: ${TIME_LIMIT}"
echo "  Memory: ${MEMORY}"
echo "  CPUs: ${CPUS}"
echo ""

JOB_ID=$(${SBATCH_CMD} \
  --time="${TIME_LIMIT}" \
  --mem="${MEMORY}" \
  --cpus-per-task="${CPUS}" \
  ${OVERRIDE_ARGS} \
  slurm_train_cartpole.sh | grep -oP '\d+$')

if [[ -n "$JOB_ID" ]]; then
  echo "✓ Job submitted successfully!"
  echo "  Job ID: ${JOB_ID}"
  echo ""
  echo "Monitor with:"
  echo "  squeue -u $USER"
  echo "  tail -f logs/oak_cartpole_${JOB_ID}.txt"
  echo ""
  echo "Cancel with:"
  echo "  scancel ${JOB_ID}"
else
  echo "✗ Job submission failed"
  exit 1
fi
