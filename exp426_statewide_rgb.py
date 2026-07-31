"""
exp426_statewide_rgb.py
========================
Statewide RGB composite inference for exp426.
Each tile outputs exp426_rgb_<tile_id>.tif (3-band uint8, R=Fibric G=Hemic B=Sapric)
Then mosaics all tiles into exp426_rgb_minnesota.tif

Usage:
    python exp426_statewide_rgb.py --tile_idx 42     # single tile (SLURM array)
    python exp426_statewide_rgb.py --mosaic          # mosaic all tiles
"""

import os, sys, json, pickle, argparse, time, glob
import numpy as np
import pandas as pd
import rasterio
import fiona
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.merge import merge
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE       = '/scratch.global/ocon0444/peat_modeling'
COV_DIR    = os.path.join(BASE, '00_data/covariates_10m')
BDY_DIR    = os.path.join(BASE, '00_data/boundary')
MDL_DIR    = os.path.join(BASE, '03_models')
OUT_DIR    = os.path.join(BASE, '04_predictions/exp426_statewide')
DEM_PATH   = os.path.join(COV_DIR, 'minnesota_dem_10m.tif')
GRID_SHP   = os.path.join(BDY_DIR, 'mn_50km_grid.shp')

EXP426_DIR = os.path.join(MDL_DIR, 'exp426')
EXP410_DIR = os.path.join(MDL_DIR, 'exp410')

N_FOLDS    = 5
PROB_THRESH= 0.33
CHUNK_SIZE = 100000
os.makedirs(OUT_DIR, exist_ok=True)

# ── PARSE ARGS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--tile_idx', type=int, default=None)
parser.add_argument('--mosaic',   action='store_true')
args = parser.parse_args()

# ── MOSAIC ─────────────────────────────────────────────────────────────────────
if args.mosaic:
    print("Mosaicking RGB statewide tiles...")
    tiles = sorted(glob.glob(os.path.join(OUT_DIR, 'exp426_rgb_tile_*.tif')))
    print(f"Found {len(tiles)} RGB tiles")
    if len(tiles) == 0:
        print("No tiles found.")
        sys.exit(1)

    missing = 115 - len(tiles)
    if missing > 0:
        print(f"WARNING: {missing} tiles missing — mosaic will have gaps")

    src_files = [rasterio.open(t) for t in tiles]
    mosaic, out_transform = merge(src_files, method='first', nodata=0)
    for src in src_files:
        src.close()

    with rasterio.open(tiles[0]) as src:
        out_profile = src.profile.copy()

    out_profile.update({
        'height':    mosaic.shape[1],
        'width':     mosaic.shape[2],
        'transform': out_transform,
        'nodata':    0,
        'dtype':     'uint8',
        'count':     3,
        'compress':  'lzw',
        'bigtiff':   'YES',
    })

    out_path = os.path.join(OUT_DIR, 'exp426_rgb_minnesota.tif')
    print(f"Writing mosaic: {mosaic.shape[1]:,} x {mosaic.shape[2]:,} pixels...")
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        for b in range(3):
            dst.write(mosaic[b], b+1)

    size_gb = os.path.getsize(out_path) / 1e9
    print(f"Saved: {out_path}  ({size_gb:.2f} GB)")
    print("RGB bands: Band1=Red=Fibric(Oi)  Band2=Green=Hemic(Oe)  Band3=Blue=Sapric(Oa)")
    sys.exit(0)

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
print("Loading models...")
with open(os.path.join(EXP426_DIR, 'feature_list.json')) as f:
    meta = json.load(f)
feature_cols = meta['features']

models_fibric     = [pickle.load(open(os.path.join(EXP426_DIR, f'model_fibric_fold_{i}.pkl'),'rb')) for i in range(N_FOLDS)]
models_hemic_frac = [pickle.load(open(os.path.join(EXP426_DIR, f'model_hemic_frac_fold_{i}.pkl'),'rb')) for i in range(N_FOLDS)]

with open(os.path.join(EXP410_DIR, 'feature_list.json')) as f:
    exp410_feats = json.load(f)['features']
exp410_models = [pickle.load(open(os.path.join(EXP410_DIR, f'model_fold_{i}.pkl'),'rb')) for i in range(N_FOLDS)]
print("  Models loaded.")

# ── RASTER DEFINITIONS ─────────────────────────────────────────────────────────
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

def read_forced(path, ref_bounds, H, W, band=1):
    with rasterio.open(path) as src:
        window = src.window(*ref_bounds)
        data = src.read(band, window=window, out_shape=(H, W),
                        resampling=Resampling.bilinear).astype(float)
        nd = src.nodata
        if nd is not None:
            data[data == nd] = np.nan
    return data

# ── SINGLE TILE ────────────────────────────────────────────────────────────────
with fiona.open(GRID_SHP) as shp:
    tiles = list(shp)

if args.tile_idx >= len(tiles):
    print(f"tile_idx {args.tile_idx} out of range")
    sys.exit(1)

tile_feat = tiles[args.tile_idx]
tile_id   = tile_feat['properties']['tile_id']
out_path  = os.path.join(OUT_DIR, f'exp426_rgb_{tile_id}.tif')

if os.path.exists(out_path):
    print(f"Already exists — skipping: {out_path}")
    sys.exit(0)

print(f"\nProcessing tile: {tile_id}")
t0 = time.time()
geoms = [tile_feat['geometry']]

try:
    with rasterio.open(DEM_PATH) as src:
        out, out_transform = rio_mask(src, geoms, crop=True, filled=True)
        ref_profile = src.profile.copy()
        H, W = out.shape[1], out.shape[2]
        ref_bounds = array_bounds(H, W, out_transform)
except Exception as e:
    print(f"ERROR masking DEM: {e}")
    sys.exit(1)

n_pixels = H * W
print(f"  Grid: {H} x {W} = {n_pixels:,} pixels")

# Load covariates
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

nwi_raw = col_arrays.get('mn_nwi_cowardin_10m', np.zeros((H,W)))
col_arrays['mn_nwi_binary']     = ((nwi_raw==1)|(nwi_raw==2)).astype(float)
col_arrays['mn_nwi_cowardin_0'] = (nwi_raw==0).astype(float)

flat       = {col: arr.ravel() for col, arr in col_arrays.items()}
dem_flat   = flat.get('minnesota_dem_10m', np.full(n_pixels, np.nan))
valid_mask = ~np.isnan(dem_flat)

if valid_mask.sum() == 0:
    print("  No valid pixels — skipping")
    sys.exit(0)

# exp410 mask
peat_prob_flat = np.zeros(n_pixels)
valid_idx = np.where(valid_mask)[0]
for chunk_start in range(0, len(valid_idx), CHUNK_SIZE):
    chunk_idx = valid_idx[chunk_start:chunk_start+CHUNK_SIZE]
    chunk_data = {col: np.where(np.isnan(flat[col][chunk_idx]),0,flat[col][chunk_idx])
                  if col in flat else np.zeros(len(chunk_idx))
                  for col in exp410_feats}
    X_chunk = pd.DataFrame(chunk_data)
    probs = np.stack([m.predict_proba(X_chunk)[:,1] for m in exp410_models]).mean(axis=0)
    peat_prob_flat[chunk_idx] = probs

peat_mask = (peat_prob_flat >= PROB_THRESH) & valid_mask
peat_idx  = np.where(peat_mask)[0]
print(f"  Peat pixels: {peat_mask.sum():,}")

if peat_mask.sum() == 0:
    print("  No peat pixels — saving blank tile")
    rgb_profile = ref_profile.copy()
    rgb_profile.update({'height':H,'width':W,'transform':out_transform,
                        'nodata':0,'dtype':'uint8','count':3,'compress':'lzw'})
    with rasterio.open(out_path,'w',**rgb_profile) as dst:
        for b in range(1,4):
            dst.write(np.zeros((H,W),dtype='uint8'), b)
    sys.exit(0)

# Predict
fibric_flat = np.zeros(n_pixels)
hemic_flat  = np.zeros(n_pixels)
sapric_flat = np.zeros(n_pixels)

for chunk_start in range(0, len(peat_idx), CHUNK_SIZE):
    chunk_idx = peat_idx[chunk_start:chunk_start+CHUNK_SIZE]
    chunk_data = {col: np.where(np.isnan(flat[col][chunk_idx]),0,flat[col][chunk_idx])
                  if col in flat else np.zeros(len(chunk_idx))
                  for col in feature_cols}
    X_chunk = pd.DataFrame(chunk_data)
    pred_fibric     = np.clip(np.stack([m.predict(X_chunk) for m in models_fibric]).mean(axis=0), 0, 100)
    pred_hemic_frac = np.clip(np.stack([m.predict(X_chunk) for m in models_hemic_frac]).mean(axis=0), 0, 1)
    remainder       = np.clip(100 - pred_fibric, 0, 100)
    fibric_flat[chunk_idx] = pred_fibric
    hemic_flat[chunk_idx]  = remainder * pred_hemic_frac
    sapric_flat[chunk_idx] = remainder * (1 - pred_hemic_frac)

# Scale to 0-255, nodata=0 outside peat mask
def scale_band(data_flat, mask):
    arr = data_flat.reshape(H, W)
    scaled = np.clip(arr / 100.0 * 255, 0, 255).astype('uint8')
    scaled[~mask.reshape(H, W)] = 0
    return scaled

r_band = scale_band(fibric_flat, peat_mask)
g_band = scale_band(hemic_flat,  peat_mask)
b_band = scale_band(sapric_flat, peat_mask)

rgb_profile = ref_profile.copy()
rgb_profile.update({
    'height': H, 'width': W, 'transform': out_transform,
    'nodata': 0, 'dtype': 'uint8', 'count': 3, 'compress': 'lzw',
})
with rasterio.open(out_path, 'w', **rgb_profile) as dst:
    dst.write(r_band, 1)
    dst.write(g_band, 2)
    dst.write(b_band, 3)

elapsed = time.time() - t0
print(f"  Saved: {out_path}  ({elapsed/60:.1f} min)")
