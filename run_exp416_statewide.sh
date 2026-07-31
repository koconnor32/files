#!/bin/bash
#SBATCH --job-name=exp416_sw
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --array=0-114
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ocon0444@umn.edu
#SBATCH -o /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/%x_%A_%a.out
#SBATCH -e /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/%x_%A_%a.err

source /users/7/ocon0444/.bashrc
conda activate gdalenvgeospat

python /scratch.global/ocon0444/peat_modeling/02_scripts/inference/exp416_statewide_inference.py \
    --tile_idx ${SLURM_ARRAY_TASK_ID}
