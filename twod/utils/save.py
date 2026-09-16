import os

# just a script to create an incremental path to save things in


def get_experiment_path(base_path="runs/exp"):
    os.makedirs("runs", exist_ok=True)
    exp_num = 0
    while os.path.exists(f"{base_path}{exp_num}"):
        exp_num += 1
    exp_path = f"{base_path}{exp_num}"
    os.makedirs(exp_path)
    return exp_path
