#!/bin/bash
#SBATCH --job-name=feat_select_exp401
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=8:00:00
#SBATCH --mail-user=ocon0444@umn.edu
#SBATCH -o /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/feat_select_exp401_%j.out
#SBATCH -e /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/feat_select_exp401_%j.err

source /users/7/ocon0444/.bashrc
conda activate gdalenvgeospat

python /scratch.global/ocon0444/peat_modeling/02_scripts/feature_selection_exp401.py
