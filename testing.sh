#!/bin/sh
#SBATCH --job-name=testing_run
#SBATCH --time=0-01:10:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ
#SBATCH --exclude=idun-11-14
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --output=logs/testing.txt      # Specifying 'stdout'


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
export PYTHONPATH="${WORKDIR}/twod:${WORKDIR}:${PYTHONPATH:-}"
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "Job started at:  $(date): "
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2025.06-1
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate
python -u twod/pipeline_testing/test_models.py \
  --threshold 0.875 \
  --save_images \
  --save_annotations \
  --per_patient \
  --no_sudden_movements \
  --threshold_sudden 20
