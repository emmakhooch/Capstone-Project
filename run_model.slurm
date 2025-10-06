#!/bin/bash
#SBATCH --job-name=food101_train       # Job name
#SBATCH --partition=gpu-research          # Partition: use gpu-a100 or gpu-v100 depending on your needs
#SBATCH --nodes=1                      # Always 1 node for GPU jobs
#SBATCH --ntasks=1                     # 1 task (process)
#SBATCH --cpus-per-task=4             # Adjust based on NUM_WORKERS in your DataLoader
#SBATCH --gres=gpu:1                   # Request 1 GPU (set to 2 if needed)
#SBATCH --mem=64G                      # Memory (adjust if you see OOM)
#SBATCH --time=24:00:00                # Walltime (hh:mm:ss)
#SBATCH --output=/mnt/nfs-scratch/ECEN_403-404/nutrician_assistant/logs/slurm-%j.out        # Stdout log (change this file name)
#SBATCH --error=/mnt/nfs-scratch/ECEN_403-404/nutrician_assistant/logs/slurm-%j.err         # Stderr log (change this file name)
#SBATCH --mail-type=ALL                # Send email when job starts/ends/fails
#SBATCH --mail-user=e_hooch@tamu.edu  # Replace with your TAMU email
#SBATCH --qos=olympus-research-gpu           #Select QOS to use

# Initialize your Miniconda install
source /mnt/nfs-scratch/ECEN_403-404/nutrician_assistant/miniconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate /mnt/nfs-scratch/ECEN_403-404/nutrician_assistant/conda_envs/food

# Navigate to your project directory
cd /mnt/nfs-scratch/ECEN_403-404/nutrician_assistant
# Run your training script
python CNN_OpenCV.py
