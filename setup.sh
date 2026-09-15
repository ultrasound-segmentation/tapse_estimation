echo "Starting setup script!"

module load Anaconda3/2025.06-1
module load Python/3.12.3-GCCcore-13.3.0
echo "Loaded Python"


if [[ ! -d ".venv" ]]; then
    echo "No virtual enviroment detected. Installing..."
    python -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

echo "Virtual enviroment has been updated."
echo "Log in to wandb..."
wandb login