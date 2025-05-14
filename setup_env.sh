#!bin/bash

# Load necessary modules
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Activate the conda environment
source activate /gpfs/wolf2/olcf/trn040/scratch/mgaber/envs/hf-transformers

# Set environment variables
export HF_HOME="/gpfs/wolf2/olcf/trn040/scratch/mgaber/hf_cache"
export HF_TOKEN=$(<hf_token.txt)