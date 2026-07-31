#!/bin/bash
#SBATCH --job-name=wbt_huc8
#SBATCH --array=1-127
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=1:00:00
#SBATCH --partition=agsmall
#SBATCH --mail-user=ocon0444@umn.edu
#SBATCH -o /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/wbt_huc8_%A_%a.out
#SBATCH -e /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/wbt_huc8_%A_%a.err

# Get list of HUC8 codes
HUC8_LIST=(/scratch.global/ocon0444/peat_modeling/00_data/dem_clipped_by_huc8/dem_huc8_*.tif)
HUC8_FILE=${HUC8_LIST[$SLURM_ARRAY_TASK_ID-1]}
HUC8_CODE=$(basename $HUC8_FILE .tif | sed 's/dem_huc8_//')

echo "Processing HUC8: $HUC8_CODE (Array task $SLURM_ARRAY_TASK_ID/127)"

/users/7/ocon0444/.conda/envs/gdalenvgeospat/bin/python \
  /scratch.global/ocon0444/peat_modeling/02_scripts/whitebox/run_wbt_flow_by_huc8.py \
  $HUC8_CODE
