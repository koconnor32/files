"""
mosaic_exp410_statewide.py
===========================
Mosaics all 115 exp410 statewide tiles into a single MN-wide raster.
Run AFTER all array jobs complete.

Output: 04_predictions/exp410_statewide/exp410_prob_minnesota.tif
"""
import os, glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import warnings
warnings.filterwarnings('ignore')

TILE_DIR = '/scratch.global/ocon0444/peat_modeling/04_predictions/exp410_statewide'
OUT_PATH = os.path.join(TILE_DIR, 'exp410_prob_minnesota.tif')

# Find all tile TIFs (exclude the mosaic itself)
tiles = sorted([f for f in glob.glob(os.path.join(TILE_DIR, 'exp410_prob_tile_*.tif'))])
print(f'Found {len(tiles)} tiles to mosaic')

if len(tiles) == 0:
    print('ERROR: No tiles found. Check tile directory.')
    exit(1)

missing = 115 - len(tiles)
if missing > 0:
    print(f'WARNING: {missing} tiles missing — mosaic will have gaps')

# Open all tiles
src_files = [rasterio.open(t) for t in tiles]

print('Mosaicking...')
mosaic, out_transform = merge(
    src_files,
    method='first',       # first valid value wins (handles buffer overlap)
    nodata=-9999.0,
    resampling=Resampling.nearest
)

# Close all sources
for src in src_files:
    src.close()

# Get profile from first tile
with rasterio.open(tiles[0]) as src:
    out_profile = src.profile.copy()

out_profile.update({
    'height':    mosaic.shape[1],
    'width':     mosaic.shape[2],
    'transform': out_transform,
    'nodata':    -9999.0,
    'dtype':     'float32',
    'count':     1,
    'compress':  'lzw',
    'bigtiff':   'YES',  # needed for large statewide raster
})

print(f'Writing mosaic: {mosaic.shape[1]:,} x {mosaic.shape[2]:,} pixels')
with rasterio.open(OUT_PATH, 'w', **out_profile) as dst:
    dst.write(mosaic[0], 1)

# Stats
valid = mosaic[0][mosaic[0] != -9999.0]
size_gb = os.path.getsize(OUT_PATH) / 1e9
print(f'\nMosaic complete:')
print(f'  File size : {size_gb:.2f} GB')
print(f'  Valid px  : {len(valid):,}')
print(f'  Mean prob : {valid.mean():.3f}')
print(f'  >0.33     : {(valid>=0.33).mean()*100:.1f}% of valid pixels')
print(f'  >0.50     : {(valid>=0.50).mean()*100:.1f}% of valid pixels')
print(f'  Saved to  : {OUT_PATH}')
