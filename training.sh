#!/bin/sh
#SBATCH --job-name=training_run
#SBATCH --time=0-00:15:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=logs/training.txt      # Specifying 'stdout'



WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
export PYTHONPATH="/cluster/home/$USER/tapse_estimation:$PYTHONPATH"
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
python twod/training.py
