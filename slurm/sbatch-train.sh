#!/bin/bash
# ==============================================================================
# Training job for HCATOstiaLocalization on cluster
# ==============================================================================
#
# SUBMIT FROM THE REPOSITORY ROOT (the folder that contains `hcatostialocalization/`):
#
#   cd /path/to/HCATOstiaLocalizationRelease
#
#   sbatch slurm/sbatch-train.sh                                  # new training run
#   sbatch slurm/sbatch-train.sh --resume-auto                    # resume latest last_model.pth
#   sbatch slurm/sbatch-train.sh --resume-checkpoint path/to/a/checkpoint.pth
#
# Everything after the script name is passed unchanged to
# `python -m hcatostialocalization.train.trainer`.
# Model / loss / hyper-parameters are chosen in
# hcatostialocalization/config/model.yaml -> active_experiment.
#
# Logs:  slurm/logs/<job-name>_<job-id>.out   (stdout)
#        slurm/logs/<job-name>_<job-id>.err   (stderr, incl. tqdm progress bars)
# Follow a running job:   tail -f slurm/logs/ostia-train_<job-id>.out
# Queue / cancel:         squeue --me        scancel <job-id>
#
# Resources are the same as the interactive session used so far:
#   srun -p jobs --qos=gpuheavy-40 --gres=gpu:40gb:1 --cpus-per-task=8 --mem=128G --time=24:00:00 --pty bash
# ==============================================================================

#SBATCH --job-name=ostia-train
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
echo " Trainer args: ${*:-<none>}"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

# ------------------------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------------------------
set +e
python -m hcatostialocalization.train.trainer "$@"
EXIT_CODE=$?
set -e

echo
echo "End        : $(date '+%Y-%m-%d %H:%M:%S')  (exit code ${EXIT_CODE})"
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[!] Training ended with an error. If it hit the time limit, resubmit with --resume-auto."
fi
exit "$EXIT_CODE"
