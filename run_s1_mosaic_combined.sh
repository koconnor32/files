#!/bin/bash
#SBATCH --job-name=s1_mosaic_combined
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/s1_mosaic_combined_%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/s1_mosaic_combined_%j.err

echo "S1 Mosaic — combined passes spring/summer/fall"
echo "Started: $(date)"

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os, glob, subprocess
import numpy as np
import rasterio
import warnings
warnings.filterwarnings('ignore')

S1_DIR  = '/scratch.global/ocon0444/S1'
OUT_DIR = '/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m'
TMP_DIR = f'{S1_DIR}/tmp'
os.makedirs(TMP_DIR, exist_ok=True)

SEASONS = ['spring', 'summer', 'fall']

def run_cmd(cmd, desc=''):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'  ERROR {desc}: {result.stderr[:200]}')
        return False
    return True

for season in SEASONS:
    out_path = f'{OUT_DIR}/minnesota_s1_{season}_combined_2019_2024.tif'
    if os.path.exists(out_path):
        print(f'{season}: already exists — skipping')
        continue

    tiles = sorted(glob.glob(f'{S1_DIR}/minnesota_s1_{season}_combined_2019_2024-*.tif'))
    print(f'\n{season}: {len(tiles)} tiles found')

    # Filter to tiles with actual data
    valid_tiles = []
    for t in tiles:
        with rasterio.open(t) as src:
            w = src.width; h = src.height
            win = rasterio.windows.Window(w//4, h//4, min(500,w//2), min(500,h//2))
            d = src.read(1, window=win)
            if (d != 0).sum() > 100:
                valid_tiles.append(t)
            else:
                # Also check center
                win2 = rasterio.windows.Window(w//2, h//2, min(500,w//2), min(500,h//2))
                d2 = src.read(1, window=win2)
                if (d2 != 0).sum() > 100:
                    valid_tiles.append(t)
    print(f'  {len(valid_tiles)}/{len(tiles)} tiles have data')

    if not valid_tiles:
        print(f'  No valid tiles for {season} — skipping')
        continue

    band_tifs = []
    for band_i, band_name in enumerate(['VV','VH','VV_VH_ratio'], start=1):
        vrt_path = f'{TMP_DIR}/{season}_b{band_i}.vrt'
        cmd = ['gdalbuildvrt','-b',str(band_i),'-vrtnodata','0',
               '-overwrite', vrt_path] + valid_tiles
        if not run_cmd(cmd, f'gdalbuildvrt {band_name}'):
            continue

        scaled = f'{TMP_DIR}/{season}_{band_name}.tif'
        cmd = ['gdal_translate','-ot','Float32',
               '-scale','-32767','32767','-327.67','327.67',
               '-a_nodata','-9999',
               '-co','COMPRESS=LZW','-co','TILED=YES','-co','BIGTIFF=YES',
               vrt_path, scaled]
        if not run_cmd(cmd, f'gdal_translate {band_name}'):
            continue

        os.remove(vrt_path)
        band_tifs.append(scaled)

        # Quick value check
        with rasterio.open(scaled) as src:
            w = src.width; h = src.height
            d = src.read(1, window=rasterio.windows.Window(w//4, h//4, 1000, 1000))
            valid = d[d != -9999]
            if len(valid) > 0:
                print(f'  {band_name}: mean={round(float(valid.mean()),2)}dB  '
                      f'range=[{round(float(valid.min()),2)},{round(float(valid.max()),2)}]')

    if len(band_tifs) != 3:
        print(f'  ERROR: only {len(band_tifs)} bands produced'); continue

    # Stack into final TIF
    stack_vrt = f'{TMP_DIR}/{season}_stack.vrt'
    run_cmd(['gdalbuildvrt','-separate','-overwrite',stack_vrt]+band_tifs, 'stack VRT')
    run_cmd(['gdal_translate','-co','COMPRESS=LZW','-co','TILED=YES',
             '-co','BIGTIFF=YES', stack_vrt, out_path], 'final translate')

    for f in band_tifs + [stack_vrt]:
        try: os.remove(f)
        except: pass

    size_gb = round(os.path.getsize(out_path)/1e9, 2)
    print(f'  Saved: {out_path} ({size_gb}GB)')

print('\nDONE')
ENDPYTHON

echo "Finished: $(date)"
