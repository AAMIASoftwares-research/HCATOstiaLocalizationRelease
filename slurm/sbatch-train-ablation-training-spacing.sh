#!/bin/bash
# ==============================================================================
# Training-spacing ablation job for HCATOstiaLocalization on cluster
# ==============================================================================
#
# SUBMIT FROM THE REPOSITORY ROOT (the folder that contains `hcatostialocalization/`):
#
#   cd /path/to/HCATOstiaLocalizationRelease
#
# The FIRST argument is the isotropic training spacing in mm (a positive number). It is passed to the
# trainer as `--spacing-iso <S>`; any further arguments are passed on unchanged.
#
#   sbatch slurm/sbatch-train-ablation-training-spacing.sh 0.75
#   sbatch slurm/sbatch-train-ablation-training-spacing.sh 0.75 --resume-auto
#   sbatch slurm/sbatch-train-ablation-training-spacing.sh 0.75 --resume-checkpoint path/to/last_model.pth
#
# One job per spacing, with a job name that tells them apart in `squeue` and in the log file names:
#   for S in 0.75 1.0 1.5; do
#       sbatch --job-name=ostia-spacing-$S slurm/sbatch-train-ablation-training-spacing.sh $S
#   done
#
# Runs `python -m hcatostialocalization.train.trainer_ablation_training_spacing`.
# Model / loss / hyper-parameters come from hcatostialocalization/config/model.yaml -> active_experiment
# (set it to swin_unetr_v2_dsnt_ablation_training_spacing for this ablation). The run is named
# <active_experiment>_ablation_training_spacing_<S>mm, which separates checkpoints and MLflow runs
# (tags: ablation=training_spacing, training_spacing_mm). --resume-auto only looks at that run's checkpoints.
#
# Logs:  slurm/logs/<job-name>_<job-id>.out   (stdout)
#        slurm/logs/<job-name>_<job-id>.err   (stderr, incl. tqdm progress bars)
# Follow a running job:   tail -f slurm/logs/<job-name>_<job-id>.out
# Queue / cancel:         squeue --me        scancel <job-id>
#
# Resources are the same as sbatch-train.sh.
# ==============================================================================

#SBATCH --job-name=ostia-spacing
#SBATCH --partition=jobs
#SBATCH --qos=gpuheavy-40
#SBATCH --gres=gpu:40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# Relative paths are resolved against the submission directory (= repository root).
# The logs folder must already exist: SLURM does not create it.
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

set -eo pipefail   # no `-u`: conda's activation scripts reference unset variables

# ------------------------------------------------------------------------------
# 1. Repository root (the trainer resolves its imports from the working directory)
# ------------------------------------------------------------------------------
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_DIR"
if [ ! -d "hcatostialocalization" ]; then
    echo "[!] '$REPO_DIR' is not the repository root (no 'hcatostialocalization/' folder)." >&2
    echo "    cd into the repository root and submit again." >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 1b. Training spacing: first argument, validated before queueing GPU time
# ------------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "[!] Missing training spacing (mm). Usage: sbatch $0 <spacing_mm> [extra trainer args]" >&2
    exit 1
fi
SPACING_ISO="$1"
shift
if ! [[ "$SPACING_ISO" =~ ^[0-9]*\.?[0-9]+$ ]] || ! awk "BEGIN{exit !($SPACING_ISO > 0)}"; then
    echo "[!] Training spacing must be a positive number (mm), got '$SPACING_ISO'." >&2
    exit 1
fi

# ------------------------------------------------------------------------------
# 2. Environment
# ------------------------------------------------------------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hcat

export PYTHONUNBUFFERED=1                      # print() appears in the .out file immediately
# Compute nodes (gn01, ...) are not listed in config/machine.yaml: use the default_user path block.
export HCAT_HOST="${HCAT_HOST:-default_user}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ------------------------------------------------------------------------------
# 3. Job summary (useful when reading old logs)
# ------------------------------------------------------------------------------
echo "=============================================================="
echo " Job        : ${SLURM_JOB_NAME:-n/a} (${SLURM_JOB_ID:-n/a})"
echo " Node       : $(hostname)"
echo " Start      : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Repo       : $REPO_DIR"
echo " Git commit : $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a') ($(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'n/a'))"
echo " Experiment : $(grep -m1 '^active_experiment:' hcatostialocalization/config/model.yaml 2>/dev/null | cut -d'"' -f2)"
echo " Python     : $(which python)"
echo " Config host: ${HCAT_HOST}"
echo " Spacing (mm): ${SPACING_ISO}"
echo " Trainer args: --spacing-iso ${SPACING_ISO} ${*}"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

# ------------------------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------------------------
set +e
python -m hcatostialocalization.train.trainer_ablation_training_spacing --spacing-iso "$SPACING_ISO" "$@"
EXIT_CODE=$?
set -e

echo
echo "End        : $(date '+%Y-%m-%d %H:%M:%S')  (exit code ${EXIT_CODE})"
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[!] Training ended with an error. If it hit the time limit, resubmit with --resume-auto."
fi
exit "$EXIT_CODE"
