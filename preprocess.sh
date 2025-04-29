#!/bin/bash

#SBATCH -A trn040
#SBATCH -J preprocessing_data
#SBATCH -o .cache/sbatch_logs/%x-%j.out
#SBATCH -e .cache/sbatch_logs/%x-%j.err
#SBATCH -t 01:00:00
#SBATCH -N 1

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a

# Activate the new environment
source activate /gpfs/wolf2/olcf/trn040/scratch/mgaber/envs/hf-transformers

/gpfs/wolf2/olcf/trn040/scratch/mgaber/envs/hf-transformers/bin/python3.12 preprocess_data.py