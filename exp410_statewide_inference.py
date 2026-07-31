"""
exp410_statewide_inference.py
==============================
Statewide peat probability inference using exp410 model.
Processes one tile at a time — designed to be called by SLURM array job.

Usage:
    python exp410_statewide_inference.py --tile_id tile_003_007
    python exp410_statewide_inference.py --tile_idx 42  (uses tile index from shapefile)

Output:
    04_predictions/exp410_statewide/exp410_prob_<tile_id>.tif
"""

import os, sys, json, pickle, argparse, time
import numpy as np
import pandas as pd
import rasterio
import fiona
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE       = '/scratch.global/ocon0444/peat_modeling'
COV_DIR    = os.path.join(BASE, '00_data/covariates_10m')
GRID_SHP   = os.path.join(BASE, '00_data/boundary/mn_50km_grid.shp')
MDL_DIR    = os.path.join(BASE, '03_models/exp410')
OUT_DIR    = os.path.join(BASE, '04_predictions/exp410_statewide')
DEM_PATH   = os.path.join(COV_DIR, 'minnesota_dem_10m.tif')
os.makedirs(OUT_DIR, exist_ok=True)

N_FOLDS    = 5
CHUNK_SIZE = 100000

# ── PARSE ARGS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--tile_id',  type=str, help='Tile ID string e.g. tile_003_007')
group.add_argument('--tile_idx', type=int, help='Tile index in shapefile (0-based, for SLURM array)')
args = parser.parse_args()

# ── LOAD TILE GEOMETRY ─────────────────────────────────────────────────────────
with fiona.open(GRID_SHP) as shp:
    tiles = list(shp)

if args.tile_idx is not None:
    if args.tile_idx >= len(tiles):
        print(f'ERROR: tile_idx {args.tile_idx} out of range (max {len(tiles)-1})')
        sys.exit(1)
    tile_feat = tiles[args.tile_idx]
    tile_id   = tile_feat['properties']['tile_id']
else:
    matching = [t for t in tiles if t['properties']['tile_id'] == args.tile_id]
    if not matching:
        print(f'ERROR: tile_id {args.tile_id} not found in shapefile')
        sys.exit(1)
    tile_feat = matching[0]
    tile_id   = args.tile_id

geom = [tile_feat['geometry']]
print(f'Processing tile: {tile_id}')
print(f'  x_min={tile_feat["properties"]["x_min"]:.0f}  '
      f'y_min={tile_feat["properties"]["y_min"]:.0f}')

# Check if output already exists
out_path = os.path.join(OUT_DIR, f'exp410_prob_{tile_id}.tif')
if os.path.exists(out_path):
    print(f'  Output already exists — skipping: {out_path}')
    sys.exit(0)

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
print('Loading exp410 models...')
with open(os.path.join(MDL_DIR, 'feature_list.json')) as f:
    meta = json.load(f)
feature_cols = meta['features']
models = [
    pickle.load(open(os.path.join(MDL_DIR, f'model_fold_{i}.pkl'), 'rb'))
    for i in range(N_FOLDS)
]
print(f'  {len(models)} fold models loaded, {len(feature_cols)} features')

# ── RASTER DEFINITIONS (must match training) ───────────────────────────────────
SENTINEL2_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","NDVI","SWDI"]
TC_BANDS        = ["TCB","TCG","TCW"]

SINGLE_BAND = [
    "minnesota_dem_10m.tif","slope.tif","aspect.tif","hillshade.tif",
    "planCurvature.tif","profileCurvature.tif","maximalCurvature.tif",
    "breached_dem.tif","d8FlowAccumulation.tif","dInfFlowAccumulation.tif",
    "wetnessIndex.tif","devfrommeanelev_4m.tif","devfrommeanelev_8m.tif",
    "devfrommeanelev_16m.tif","diffFromMeanElev.tif",
    "relativeTopographicPosition_4m.tif","relativeTopographicPosition_8m.tif",
    "relativeTopographicPosition_16m.tif","dist_to_water_10m.tif",
    "dist_to_stream_10m.tif","dist_to_road_detailed_10m.tif",
    "prism_ppt_mn.tif","prism_tmax_july_mn.tif","prism_tmean_mn.tif",
    "prism_tmin_january_mn.tif","mn_nwi_cowardin_10m.tif",
    "histosols_10m_snapped.tif","MN_ANY_organic_component_snapped.tif",
    "MN_organic_soils_classified_FIXED_snapped.tif",
    "gNATSGO_MN_26915.tif","npc_peatland_indicator_10m.tif",
]
MULTIBAND = [
    ("s2_spring_12bands.tif",     "s2_spring",   SENTINEL2_BANDS),
    ("s2_summer_12bands.tif",     "s2_summer",   SENTINEL2_BANDS),
    ("s2_fall_12bands.tif",       "s2_fall",     SENTINEL2_BANDS),
    ("tc_spring_merged_5070.tif", "tc_spring",   TC_BANDS),
    ("tc_summer_merged_5070.tif", "tc_summer",   TC_BANDS),
    ("tc_fall_merged_5070.tif",   "tc_fall",     TC_BANDS),
]

# ── HELPER ─────────────────────────────────────────────────────────────────────
def read_forced(path, ref_bounds, H, W, band=1):
    with rasterio.open(path) as src:
        window = src.window(*ref_bounds)
        data = src.read(band, window=window, out_shape=(H, W),
                        resampling=Resampling.bilinear).astype(float)
        nd = src.nodata
        if nd is not None:
            data[data == nd] = np.nan
    return data

# ── GET REFERENCE GRID ─────────────────────────────────────────────────────────
t0 = time.time()
try:
    with rasterio.open(DEM_PATH) as src:
        out, out_transform = rio_mask(src, geom, crop=True, filled=True)
        ref_profile = src.profile.copy()
        H, W = out.shape[1], out.shape[2]
        ref_bounds = array_bounds(H, W, out_transform)
except Exception as e:
    print(f'ERROR masking DEM for tile {tile_id}: {e}')
    sys.exit(1)

n_pixels = H * W
print(f'  Grid: {H} x {W} = {n_pixels:,} pixels')

if n_pixels == 0:
    print(f'  Empty tile — skipping')
    sys.exit(0)

# ── LOAD COVARIATES ────────────────────────────────────────────────────────────
print('  Loading covariates...')
col_arrays = {}

for fname in SINGLE_BAND:
    col  = fname.replace('.tif','')
    path = os.path.join(COV_DIR, fname)
    col_arrays[col] = read_forced(path, ref_bounds, H, W) if os.path.exists(path) else np.full((H,W), np.nan)

for fname, prefix, band_names in MULTIBAND:
    path = os.path.join(COV_DIR, fname)
    for i, bname in enumerate(band_names, start=1):
        col = f"{prefix}_{bname}"
        col_arrays[col] = read_forced(path, ref_bounds, H, W, band=i) if os.path.exists(path) else np.full((H,W), np.nan)

# Derived NWI columns (exp410 was trained with these)
nwi_raw = col_arrays.get('mn_nwi_cowardin_10m', np.zeros((H,W)))
col_arrays['mn_nwi_binary']     = ((nwi_raw==1)|(nwi_raw==2)).astype(float)
col_arrays['mn_nwi_cowardin_0'] = (nwi_raw==0).astype(float)

# Flatten to 1D
flat = {col: arr.ravel() for col, arr in col_arrays.items()}

# Valid pixel mask (not NaN in DEM)
dem_flat   = flat.get('minnesota_dem_10m', np.full(n_pixels, np.nan))
valid_mask = ~np.isnan(dem_flat)
print(f'  Valid pixels: {valid_mask.sum():,} / {n_pixels:,} ({100*valid_mask.mean():.1f}%)')

if valid_mask.sum() == 0:
    print(f'  No valid pixels — skipping tile')
    sys.exit(0)

# ── INFERENCE ──────────────────────────────────────────────────────────────────
print('  Running inference...')
prob_flat  = np.full(n_pixels, np.nan)
valid_idx  = np.where(valid_mask)[0]

# Check all required features are available
missing_feats = [c for c in feature_cols if c not in flat]
if missing_feats:
    print(f'  WARNING: {len(missing_feats)} features missing, filling 0: {missing_feats}')

for chunk_start in range(0, len(valid_idx), CHUNK_SIZE):
    chunk_idx = valid_idx[chunk_start:chunk_start+CHUNK_SIZE]
    chunk_data = {}
    for col in feature_cols:
        if col in flat:
            vals = flat[col][chunk_idx]
            vals = np.where(np.isnan(vals), 0, vals)
        else:
            vals = np.zeros(len(chunk_idx))
        chunk_data[col] = vals
    X_chunk = pd.DataFrame(chunk_data)
    probs = np.stack([m.predict_proba(X_chunk)[:,1] for m in models]).mean(axis=0)
    prob_flat[chunk_idx] = probs

# ── SAVE ───────────────────────────────────────────────────────────────────────
prob_2d = prob_flat.reshape(H, W).astype('float32')
prob_2d[~valid_mask.reshape(H, W)] = -9999.0

out_profile = ref_profile.copy()
out_profile.update({
    'height':    H,
    'width':     W,
    'transform': out_transform,
    'nodata':    -9999.0,
    'dtype':     'float32',
    'count':     1,
    'compress':  'lzw',
})

with rasterio.open(out_path, 'w', **out_profile) as dst:
    dst.write(prob_2d, 1)

valid_probs = prob_flat[valid_mask]
elapsed = time.time() - t0
print(f'  Done in {elapsed/60:.1f} min')
print(f'  Prob stats: mean={valid_probs.mean():.3f}  '
      f'max={valid_probs.max():.3f}  '
      f'>0.33: {(valid_probs>=0.33).mean()*100:.1f}%  '
      f'>0.5: {(valid_probs>=0.5).mean()*100:.1f}%')
print(f'  Saved: {out_path}')
