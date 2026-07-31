#!/bin/bash
#SBATCH --job-name=align_dem
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=1:00:00
#SBATCH --partition=agsmall
#SBATCH --mail-user=ocon0444@umn.edu
#SBATCH -o /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/align_dem_%j.out
#SBATCH -e /scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/align_dem_%j.err

COVAR_DIR="/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m"

echo "Aligning DEM to gNATSGO grid using -ts to force exact size..."

gdalwarp \
  -ts 66474 75185 \
  -te -99098.0 2269539.0 565642.0 3021389.0 \
  -r bilinear \
  -co COMPRESS=LZW \
  -co TILED=YES \
  -co BIGTIFF=YES \
  ${COVAR_DIR}/minnesota_dem_10m.tif \
  ${COVAR_DIR}/minnesota_dem_10m_ALIGNED.tif

echo ""
echo "Verifying..."
gdalinfo ${COVAR_DIR}/minnesota_dem_10m_ALIGNED.tif | grep "Size\|Origin"
