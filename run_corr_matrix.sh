#!/bin/bash
#SBATCH --job-name=corr_matrix
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --mail-user=ocon0444@umn.edu
#SBATCH -o /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/corr_matrix_%j.out
#SBATCH -e /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/corr_matrix_%j.err

source /users/7/ocon0444/.bashrc
conda activate gdalenvgeospat

python /scratch.global/ocon0444/peat_modeling/02_scripts/corr_matrix_covariates.py
