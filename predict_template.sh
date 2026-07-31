#!/bin/bash
#SBATCH --job-name=predict_EXP_ID_RESm
#SBATCH --ntasks=1
#SBATCH --mem=96gb
#SBATCH --time=3:00:00
#SBATCH --partition=agsmall
#SBATCH --mail-user=ocon0444@umn.edu
#SBATCH -o /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/predict_EXP_ID_RESm_%j.out
#SBATCH -e /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/predict_EXP_ID_RESm_%j.err

cd /scratch.global/ocon0444/peat_modeling/02_scripts/inference

# Replace EXP_ID and RES with your values
/users/7/ocon0444/.conda/envs/gdalenvgeospat/bin/python predict_raster.py \
  --exp_id EXP_ID \
  --resolution RES \
  --ensemble
