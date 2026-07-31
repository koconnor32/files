#!/bin/bash
#SBATCH --job-name=predict_exp201_10m
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96gb
#SBATCH --time=04:00:00
#SBATCH --partition=agsmall
#SBATCH --mail-user=ocon0444@umn.edu
#SBATCH -o /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/predict_exp201_10m_%j.out
#SBATCH -e /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/predict_exp201_10m_%j.err

cd /scratch.global/ocon0444/peat_modeling/02_scripts/inference

/users/7/ocon0444/.conda/envs/gdalenvgeospat/bin/python predict_raster_binary.py \
  --exp_id exp201 \
  --resolution 10 \
  --ensemble

