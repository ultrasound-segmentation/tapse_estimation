#!/bin/sh
#SBATCH --job-name=validation_run
#SBATCH --time=0-01:10:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1             # Setting the number of GPUs to 1
#SBATCH --constraint="a100"     # Defining the GPU type to be used
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=logs/validation.txt      # Specifying 'stdout'



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
python twod/pipeline_testing/auto_idx_calculation/statystical_analysis.py --automatic_path "data/patients.xlsx"