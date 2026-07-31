#!/bin/bash
#SBATCH --job-name=agc_jenkins_raster
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=4:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/agc_jenkins_raster_%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/agc_jenkins_raster_%j.err

echo "=============================="
echo "Jenkins AGC Rasterization + Above/Below Overlay"
echo "Started: $(date)"
echo "=============================="

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os
import numpy as np
import fiona
import rasterio
from rasterio.features import rasterize as rio_rasterize
from rasterio.mask import mask as rio_mask
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

BASE_M   = '/scratch.global/ocon0444/peat_modeling'
SHP_5070 = f'{BASE_M}/00_data/dnr_forest_stand_inventory_jenkins_5070.shp'
REF_RAS  = f'{BASE_M}/04_predictions/exp410_statewide/exp410_prob_minnesota.tif'
AGC_OUT  = f'{BASE_M}/04_predictions/exp_carbon_stock/agc_jenkins_forest.tif'
OVL_DIR  = f'{BASE_M}/04_predictions/exp_carbon_stock/above_below_overlay'
BG_DIR   = f'{BASE_M}/04_predictions/exp_carbon_stock'
os.makedirs(OVL_DIR, exist_ok=True)

CRS = rasterio.crs.CRS.from_epsg(5070)

REGIONS = ['RedLake', 'SE', 'SW']

# ── Step 1: Load forest stand geometries and carbon values ─────────
print('Loading Jenkins AGC shapefile...')
with fiona.open(SHP_5070) as src:
    print(f'  CRS: {src.crs}')
    print(f'  Features: {len(src):,}')
    fields = list(src.schema['properties'].keys())
    # Handle truncated field name
    agc_field = 'C_AGC_KGM2' if 'C_AGC_KGM2' in fields else \
                next((f for f in fields if 'AGC' in f.upper()), None)
    print(f'  Carbon field: {agc_field}')

    geom_val_pairs = []
    n_skip = 0
    for feat in src:
        agc = feat['properties'].get(agc_field, 0) or 0
        if agc > 0 and feat['geometry'] is not None:
            geom_val_pairs.append((feat['geometry'], float(agc)))
        else:
            n_skip += 1

print(f'  Forested stands: {len(geom_val_pairs):,}')
print(f'  Skipped (no carbon): {n_skip:,}')

# ── Step 2: Rasterize tile by tile to avoid OOM ────────────────────
print('\nRasterizing AGC to statewide 10m grid (tile by tile)...')

with rasterio.open(REF_RAS) as ref:
    ref_transform = ref.transform
    ref_profile   = ref.profile.copy()
    nrows         = ref.height
    ncols         = ref.width
    res           = ref.res[0]

print(f'  Grid: {nrows} x {ncols} pixels')

out_profile = ref_profile.copy()
out_profile.update(dtype='float32', count=1, nodata=-9999,
                   compress='lzw', bigtiff='YES')

TILE_PX = 5000  # 50km tiles

# Initialize output
print('  Initializing output raster...')
with rasterio.open(AGC_OUT, 'w', **out_profile) as dst:
    for r in range(0, nrows, 1000):
        h_chunk = min(1000, nrows - r)
        block   = np.full((h_chunk, ncols), -9999, dtype=np.float32)
        dst.write(block, 1, window=Window(0, r, ncols, h_chunk))
print('  Initialized.')

# Rasterize tile by tile
n_tiles_done = 0
with rasterio.open(AGC_OUT, 'r+') as dst:
    for row_off in range(0, nrows, TILE_PX):
        for col_off in range(0, ncols, TILE_PX):
            h = min(TILE_PX, nrows - row_off)
            w = min(TILE_PX, ncols - col_off)
            window = Window(col_off, row_off, w, h)
            win_tf = rasterio.windows.transform(window, ref_transform)

            tile = rio_rasterize(
                geom_val_pairs,
                out_shape=(h, w),
                transform=win_tf,
                fill=-9999,
                dtype=np.float32,
            )
            dst.write(tile, 1, window=window)
            n_tiles_done += 1
            if n_tiles_done % 20 == 0:
                valid = tile[tile > 0]
                print(f'  Tile {n_tiles_done} ({row_off},{col_off})  '
                      f'forested px={len(valid):,}')

# Quick stats
print('\nAGC raster stats (sampling every 100 rows)...')
vals = []
with rasterio.open(AGC_OUT) as src:
    for r in range(0, src.height, 100):
        row = src.read(1, window=Window(0, r, src.width, 1)).flatten()
        vals.extend(row[row > 0].tolist())
vals = np.array(vals)
if len(vals) > 0:
    print(f'  Valid pixels (sample): {len(vals):,}')
    print(f'  Mean: {vals.mean():.3f} kgC/m2  ({vals.mean()*10:.1f} MgC/ha)')
    print(f'  Range: [{vals.min():.3f}, {vals.max():.3f}]')
print(f'  Saved: {AGC_OUT}')

# ── Step 3: Above/below overlay per region ─────────────────────────
print('\n' + '='*60)
print('Computing above/below carbon overlay for 3 regions...')
print('='*60)

def save_raster(arr, path, ref_profile):
    prof = ref_profile.copy()
    prof.update(dtype='float32', count=1, nodata=-9999, compress='lzw')
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(arr.astype(np.float32), 1)
    size = os.path.getsize(path)/1e6
    print(f'    Saved: {os.path.basename(path)}  ({size:.1f} MB)')

for region_name in REGIONS:
    print(f'\n--- {region_name} ---')
    shp_path  = f'{BASE_M}/00_data/boundary/sample_boundary_{region_name}.shp'
    bg_path   = f'{BG_DIR}/carbon_full_stock_{region_name}.tif'

    with fiona.open(shp_path) as src:
        geoms = [shape(f['geometry']) for f in src]
    region_geom = unary_union(geoms)
    geom_json   = [region_geom.__geo_interface__]

    # Read belowground as reference grid
    with rasterio.open(bg_path) as src:
        bg_data, ref_tf = rio_mask(src, geom_json, crop=True)
        bg_arr  = bg_data.squeeze().astype(np.float32)
        nd      = src.nodata or -9999
        bg_arr[bg_arr == nd] = np.nan
        ref_profile_reg = src.profile.copy()
        ref_profile_reg.update(transform=ref_tf,
                               height=bg_arr.shape[0],
                               width=bg_arr.shape[1])
    h, w = bg_arr.shape
    print(f'  Grid: {h} x {w}')

    # Read AGC raster clipped to region
    with rasterio.open(AGC_OUT) as src:
        ag_data, _ = rio_mask(src, geom_json, crop=True)
        ag_arr = ag_data.squeeze().astype(np.float32)
        nd_ag  = src.nodata or -9999
        ag_arr[ag_arr == nd_ag] = np.nan

    # Resize if shapes don't match exactly
    if ag_arr.shape != (h, w):
        from skimage.transform import resize
        ag_arr = resize(ag_arr, (h, w), order=0,
                        preserve_range=True, anti_aliasing=False).astype(np.float32)

    has_bg = ~np.isnan(bg_arr)
    has_ag = (ag_arr > 0) & ~np.isnan(ag_arr)
    ag_safe = np.where(np.isnan(ag_arr) | (ag_arr < 0), 0, ag_arr)
    bg_safe = np.where(np.isnan(bg_arr), 0, bg_arr)

    # Total carbon — only where at least one exists
    total_arr = np.full((h, w), np.nan, dtype=np.float32)
    both      = has_bg & has_ag
    bg_only   = has_bg & ~has_ag
    ag_only   = ~has_bg & has_ag
    total_arr[both]    = bg_arr[both] + ag_arr[both]
    total_arr[bg_only] = bg_arr[bg_only]
    total_arr[ag_only] = ag_arr[ag_only]

    print(f'  Both above+below: {both.sum():,}')
    print(f'  Below only:       {bg_only.sum():,}')
    print(f'  Above only:       {ag_only.sum():,}')

    # Percent above and below
    pct_ag = np.full((h, w), np.nan, dtype=np.float32)
    pct_bg = np.full((h, w), np.nan, dtype=np.float32)
    valid  = ~np.isnan(total_arr) & (total_arr > 0)
    pct_ag[valid] = (ag_safe[valid] / total_arr[valid]) * 100
    pct_bg[valid] = (bg_safe[valid] / total_arr[valid]) * 100

    print(f'  Mean % aboveground: {np.nanmean(pct_ag):.1f}%')
    print(f'  Mean % belowground: {np.nanmean(pct_bg):.1f}%')

    def to_nodata(arr):
        return np.where(np.isnan(arr), -9999, arr)

    save_raster(to_nodata(ag_arr),    f'{OVL_DIR}/ag_carbon_{region_name}.tif',    ref_profile_reg)
    save_raster(to_nodata(bg_arr),    f'{OVL_DIR}/bg_carbon_{region_name}.tif',    ref_profile_reg)
    save_raster(to_nodata(total_arr), f'{OVL_DIR}/total_carbon_{region_name}.tif', ref_profile_reg)
    save_raster(to_nodata(pct_ag),    f'{OVL_DIR}/pct_ag_{region_name}.tif',       ref_profile_reg)
    save_raster(to_nodata(pct_bg),    f'{OVL_DIR}/pct_bg_{region_name}.tif',       ref_profile_reg)

print('\n' + '='*60)
print('DONE.')
print(f'AGC raster:      {AGC_OUT}')
print(f'Overlay rasters: {OVL_DIR}/')
print('='*60)
ENDPYTHON

echo ""
echo "=============================="
echo "Finished: $(date)"
echo "=============================="
