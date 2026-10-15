#!/bin/bash
# ==============================================================================
# ImageCAS training-set-size ablation job for HCATOstiaLocalization on cluster
# ==============================================================================
#
# SUBMIT FROM THE REPOSITORY ROOT (the folder that contains `hcatostialocalization/`):
#
#   cd /path/to/HCATOstiaLocalizationRelease
#
# The FIRST argument is the number of ImageCAS cases (train+val combined): an integer >= 0 or `all`.
# It is passed to the trainer as `--imagecas-n <N>`; any further arguments are passed on unchanged.
#
#   sbatch slurm/sbatch-train-ablation-imagecas-size.sh 250
#   sbatch slurm/sbatch-train-ablation-imagecas-size.sh 250 --resume-auto
#   sbatch slurm/sbatch-train-ablation-imagecas-size.sh 250 --resume-checkpoint path/to/last_model.pth
#
# One job per N, with a job name that tells them apart in `squeue` and in the log file names:
#   for N in 0 250 500 750 all; do
#       sbatch --job-name=ostia-imagecas-$N slurm/sbatch-train-ablation-imagecas-size.sh $N
#   done
#
# Runs `python -m hcatostialocalization.train.trainer_ablation_imagecas_size`.
# Model / loss / hyper-parameters come from hcatostialocalization/config/model.yaml -> active_experiment
# (set it to swin_unetr_v2_dsnt for this ablation). The run is named
# <active_experiment>_ablation_training_set_size_<N>, which separates checkpoints and MLflow runs.
#
# Logs:  slurm/logs/<job-name>_<job-id>.out   (stdout)
#        slurm/logs/<job-name>_<job-id>.err   (stderr, incl. tqdm progress bars)
# Follow a running job:   tail -f slurm/logs/<job-name>_<job-id>.out
# Queue / cancel:         squeue --me        scancel <job-id>
#
# Resources are the same as sbatch-train.sh.
# ==============================================================================

#SBATCH --job-name=ostia-imagecas-n
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
# 1b. ImageCAS subset size: first argument, validated before queueing GPU time
# ------------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "[!] Missing ImageCAS size. Usage: sbatch $0 <N|all> [extra trainer args]" >&2
    exit 1
fi
IMAGECAS_N="$1"
shift
if ! [[ "$IMAGECAS_N" =~ ^[0-9]+$ || "$IMAGECAS_N" == "all" ]]; then
    echo "[!] ImageCAS size must be a non-negative integer or 'all', got '$IMAGECAS_N'." >&2
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
echo " ImageCAS N : ${IMAGECAS_N}"
echo " Trainer args: --imagecas-n ${IMAGECAS_N} ${*}"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo

# ------------------------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------------------------
set +e
python -m hcatostialocalization.train.trainer_ablation_imagecas_size --imagecas-n "$IMAGECAS_N" "$@"
EXIT_CODE=$?
set -e

echo
echo "End        : $(date '+%Y-%m-%d %H:%M:%S')  (exit code ${EXIT_CODE})"
if [ "$EXIT_CODE" -ne 0 ]; then
    echo "[!] Training ended with an error. If it hit the time limit, resubmit with --resume-auto."
fi
exit "$EXIT_CODE"
