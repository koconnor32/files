#!/usr/bin/env python3
"""
Tasseled Cap for Sentinel-2 — clean rewrite with verification at every step.
Writes 3 SEPARATE tifs per season (safer than modifying 15-band files):
  s2_spring_TCB.tif, s2_spring_TCG.tif, s2_spring_TCW.tif
  s2_summer_TCB.tif, s2_summer_TCG.tif, s2_summer_TCW.tif
  s2_fall_TCB.tif,   s2_fall_TCG.tif,   s2_fall_TCW.tif

Tasseled Cap coefficients: Nedkov 2017, Sentinel-2 surface reflectance
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import os, argparse

RASTER_DIR = '/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m'

# Nedkov 2017 TC coefficients for S2
# Bands: B2=blue B3=green B4=red B8=NIR B11=SWIR1 B12=SWIR2
TC = {
    'TCB': [ 0.3029,  0.2786,  0.4733,  0.5599,  0.5080,  0.1872],
    'TCG': [-0.2941, -0.2430, -0.5424,  0.7276,  0.0713, -0.1608],
    'TCW': [ 0.1511,  0.1973,  0.3283,  0.3407, -0.7117, -0.4559],
}

# Band indices in 12-band file (1-indexed): B02 B03 B04 B05 B06 B07 B08 B8A B11 B12 NDVI NDMI
# We need:  B2=1  B3=2  B4=3  B8=7  B11=9  B12=10
TC_BAND_IDX = [1, 2, 3, 7, 9, 10]
TC_BAND_NAMES = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']

CHUNK = 512

def verify_input(src, season):
    """Check input bands have real data."""
    print(f'\n  Input verification ({season}_12bands):')
    w = Window(src.width//3, src.height//3, 300, 300)
    for i, (idx, name) in enumerate(zip(TC_BAND_IDX, TC_BAND_NAMES)):
        data = src.read(idx, window=w).astype(np.float32).flatten()
        nonzero = (data != 0).sum()
        print(f'    Band {idx} ({name}): min={data.min():.0f}  max={data.max():.0f}  '
              f'mean={data.mean():.0f}  nonzero={nonzero}/{len(data)}')

def process_season(season, tc_name, coeff):
    input_path  = f'{RASTER_DIR}/s2_{season}_12bands.tif'
    output_path = f'{RASTER_DIR}/s2_{season}_{tc_name}.tif'

    if not os.path.exists(input_path):
        print(f'  SKIP: {input_path} not found')
        return False

    if os.path.exists(output_path):
        print(f'  SKIP: {output_path} already exists — delete to rerun')
        return True

    print(f'\n  Processing {season} {tc_name}...')

    with rasterio.open(input_path) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='float32',
                    compress='lzw', tiled=True, bigtiff='YES')

        with rasterio.open(output_path, 'w', **meta) as dst:
            total_chunks = (src.height // CHUNK + 1) * (src.width // CHUNK + 1)
            done = 0
            for row in range(0, src.height, CHUNK):
                for col in range(0, src.width, CHUNK):
                    w = Window(col, row,
                               min(CHUNK, src.width  - col),
                               min(CHUNK, src.height - row))

                    # Read 6 bands needed for TC
                    result = np.zeros(
                        (min(CHUNK, src.height-row), min(CHUNK, src.width-col)),
                        dtype=np.float32)

                    for band_idx, c in zip(TC_BAND_IDX, coeff):
                        band_data = src.read(band_idx, window=w).astype(np.float32)
                        result += c * band_data

                    dst.write(result, 1, window=w)
                    done += 1
                    if done % 200 == 0:
                        pct = done / total_chunks * 100
                        print(f'    {pct:.0f}% complete...', flush=True)

    # Verify output
    with rasterio.open(output_path) as src:
        w = Window(src.width//3, src.height//3, 300, 300)
        data = src.read(1, window=w).flatten()
        nonzero = (data != 0).sum()
        print(f'  Output {tc_name}: min={data.min():.1f}  max={data.max():.1f}  '
              f'mean={data.mean():.1f}  nonzero={nonzero}/{len(data)}')
        if nonzero == 0:
            print(f'  WARNING: all zeros in sample window — check input bands!')
            return False

    size_gb = os.path.getsize(output_path) / 1e9
    print(f'  Saved: {output_path} ({size_gb:.1f} GB)')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+',
                        choices=['spring','summer','fall'],
                        default=['spring','summer'])
    args = parser.parse_args()

    # First verify input bands look real
    print('='*60)
    print('STEP 1: Verify input band values')
    print('='*60)
    for season in args.seasons:
        path = f'{RASTER_DIR}/s2_{season}_12bands.tif'
        if os.path.exists(path):
            with rasterio.open(path) as src:
                verify_input(src, season)

    print('\n' + '='*60)
    print('STEP 2: Calculate Tasseled Cap bands')
    print('='*60)

    for season in args.seasons:
        print(f'\nSeason: {season}')
        for tc_name, coeff in TC.items():
            ok = process_season(season, tc_name, coeff)
            if not ok:
                print(f'  FAILED: {season} {tc_name}')

    print('\n' + '='*60)
    print('DONE — output files:')
    for season in args.seasons:
        for tc_name in TC:
            path = f'{RASTER_DIR}/s2_{season}_{tc_name}.tif'
            if os.path.exists(path):
                gb = os.path.getsize(path) / 1e9
                print(f'  {path} ({gb:.1f} GB)')
    print('='*60)


if __name__ == '__main__':
    main()
