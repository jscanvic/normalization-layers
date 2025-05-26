import subprocess

script = f"""\
#!/bin/sh
#SBATCH --job-name=Job
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --constraint=h100

module purge

module load miniforge/24.9.0
conda activate "$HOME/env"

cd "$HOME/wd"

set -x
srun python -u "$HOME/code/main.py" --data_root "$HOME/datasets/ImageNet" --out "$HOME/results.csv" --batch_size 1024
"""
process = subprocess.run(["sbatch"], input=script, text=True, check=True)
