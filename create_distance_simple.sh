#!/bin/bash
#SBATCH --job-name=regen_wbt
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --partition=agsmall
#SBATCH -o /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/regen_wbt_%j.out
#SBATCH -e /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/regen_wbt_%j.err

source /users/7/ocon0444/.bashrc
conda activate gdalenvgeospat

python /scratch.global/ocon0444/peat_modeling/02_scripts/whitebox/regenerate_broken_wbt.py
