"""
make_cogs_exp426.py
====================
Mosaics all exp426 statewide tiles (fibric, hemic, sapric, dominant, rgb)
and writes COG files for each.

Usage:
    python make_cogs_exp426.py
"""
import os, glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import warnings
warnings.filterwarnings('ignore')

BASE    = '/scratch.global/ocon0444/peat_modeling'
OUT_DIR = os.path.join(BASE, '04_predictions/exp426_statewide')

RASTER_TYPES = [
    ('fibric',   'float32', -9999.0, 1),
    ('hemic',    'float32', -9999.0, 1),
    ('sapric',   'float32', -9999.0, 1),
    ('dominant', 'float32', -9999.0, 1),
    ('rgb',      'uint8',   0,       3),
]

for rtype, dtype, nodata, n_bands in RASTER_TYPES:
    print(f'\n{"="*55}')
    print(f'Processing: {rtype}')
    print('='*55)

    tiles = sorted(glob.glob(os.path.join(OUT_DIR, f'exp426_{rtype}_tile_*.tif')))
    print(f'  Found {len(tiles)} tiles')
    if len(tiles) == 0:
        print(f'  No tiles found — skipping')
        continue
    if len(tiles) < 115:
        print(f'  WARNING: only {len(tiles)}/115 tiles')

    MOSAIC_PATH = os.path.join(OUT_DIR, f'exp426_{rtype}_minnesota.tif')
    COG_PATH    = os.path.join(OUT_DIR, f'exp426_{rtype}_minnesota_COG.tif')

    # ── MOSAIC ────────────────────────────────────────────────────────────────
    print('  Merging...')
    src_files = [rasterio.open(t) for t in tiles]
    mosaic, transform = merge(src_files, method='first', nodata=nodata)
    profile = src_files[0].profile.copy()
    for s in src_files:
        s.close()

    profile.update({
        'height':    mosaic.shape[1],
        'width':     mosaic.shape[2],
        'transform': transform,
        'nodata':    nodata,
        'dtype':     dtype,
        'count':     n_bands,
        'compress':  'deflate',
        'predictor': 2,
        'tiled':     True,
        'blockxsize':512,
        'blockysize':512,
        'bigtiff':   'YES',
    })

    with rasterio.open(MOSAIC_PATH, 'w', **profile) as dst:
        if n_bands == 1:
            dst.write(mosaic[0], 1)
        else:
            for b in range(n_bands):
                dst.write(mosaic[b], b+1)

    size_gb = os.path.getsize(MOSAIC_PATH) / 1e9
    print(f'  Mosaic: {MOSAIC_PATH}  ({size_gb:.2f} GB)')

    # Quick stats
    if n_bands == 1:
        valid = mosaic[0][mosaic[0] != nodata]
        if len(valid) > 0:
            if rtype == 'dominant':
                for i, label in [(1,'Fibric'),(2,'Hemic'),(3,'Sapric')]:
                    pct = 100*(valid==i).mean()
                    print(f'    {label}: {pct:.1f}%')
            else:
                print(f'    mean={valid.mean():.1f}  min={valid.min():.1f}  max={valid.max():.1f}')

    # ── OVERVIEWS + COG ───────────────────────────────────────────────────────
    print('  Building overviews...')
    resampling = Resampling.nearest if rtype == 'dominant' else Resampling.average
    with rasterio.open(MOSAIC_PATH, 'r+') as src:
        src.build_overviews([2,4,8,16,32,64], resampling)
        src.update_tags(ns='rio_overview', resampling=resampling.name)

    print('  Writing COG...')
    with rasterio.open(MOSAIC_PATH) as src:
        cog_profile = src.profile.copy()
        cog_profile.update(copy_src_overviews=True)
        with rasterio.open(COG_PATH, 'w', **cog_profile) as dst:
            dst.write(src.read())

    size_cog = os.path.getsize(COG_PATH) / 1e9
    print(f'  COG: {COG_PATH}  ({size_cog:.2f} GB)')

print('\nAll exp426 COGs complete.')
