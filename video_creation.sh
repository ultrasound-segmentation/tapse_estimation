#!/bin/sh
#SBATCH --job-name=video_creation_run
#SBATCH --time=0-00:20:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --output=logs/video_creation.txt

WORKDIR=${SLURM_SUBMIT_DIR}
cd "${WORKDIR}" || exit 1
export PYTHONPATH="${WORKDIR}/twod:${WORKDIR}:${PYTHONPATH:-}"
echo "Running from this directory: ${SLURM_SUBMIT_DIR}"
echo "Name of job: ${SLURM_JOB_NAME}"
echo "Job started at:  $(date)"
echo "ID of job: ${SLURM_JOB_ID}"
echo "The job was run on these nodes: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2025.06-1
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

python -u twod/pipeline_testing/visual_analysis/video_illustrative.py
