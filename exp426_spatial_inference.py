"""
exp426_spatial_inference.py
============================
Spatial inference for exp426 — Compositional RF (iterative residual)
Fibric + Hemic + Sapric = 100% by construction at every pixel.

Part 1 — Regional tiles (SE, SW, RedLake):
  Outputs per region:
    exp426_fibric_<region>.tif
    exp426_hemic_<region>.tif
    exp426_sapric_<region>.tif
    exp426_rgb_<region>.tif       (RGB composite: R=Fibric G=Hemic B=Sapric)

Part 2 — Statewide dominant type (run separately via --statewide flag):
  Per tile: exp426_dominant_tile_XXX_YYY.tif  (1=Fibric 2=Hemic 3=Sapric)
  Mosaic:   exp426_dominant_minnesota.tif

Usage:
    python exp426_spatial_inference.py                  # regional tiles
    python exp426_spatial_inference.py --statewide      # all 115 tiles
    python exp426_spatial_inference.py --tile_idx 42    # single tile (SLURM array)
    python exp426_spatial_inference.py --mosaic         # mosaic dominant tiles
"""

import os, sys, json, pickle, argparse, time
import numpy as np
import pandas as pd
import rasterio
import fiona
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE       = '/scratch.global/ocon0444/peat_modeling'
COV_DIR    = os.path.join(BASE, '00_data/covariates_10m')
BDY_DIR    = os.path.join(BASE, '00_data/boundary')
MDL_DIR    = os.path.join(BASE, '03_models')
OUT_REG    = os.path.join(BASE, '04_predictions/exp426')
OUT_SW     = os.path.join(BASE, '04_predictions/exp426_statewide')
DEM_PATH   = os.path.join(COV_DIR, 'minnesota_dem_10m.tif')
GRID_SHP   = os.path.join(BDY_DIR, 'mn_50km_grid.shp')

EXP426_DIR = os.path.join(MDL_DIR, 'exp426')
EXP410_DIR = os.path.join(MDL_DIR, 'exp410')

N_FOLDS    = 5
PROB_THRESH= 0.33
CHUNK_SIZE = 100000
DOMINANT_LABELS = {1: 'Fibric(Oi)', 2: 'Hemic(Oe)', 3: 'Sapric(Oa)'}

os.makedirs(OUT_REG, exist_ok=True)
os.makedirs(OUT_SW,  exist_ok=True)

REGIONAL_BOUNDARIES = {
    'redlake': os.path.join(BDY_DIR, 'sample_boundary_RedLake.shp'),
    'SE':      os.path.join(BDY_DIR, 'sample_boundary_SE.shp'),
    'SW':      os.path.join(BDY_DIR, 'sample_boundary_SW.shp'),
}

# ── PARSE ARGS ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--statewide',  action='store_true', help='Run all 115 tiles')
parser.add_argument('--tile_idx',   type=int, default=None, help='Single tile index (SLURM array)')
parser.add_argument('--mosaic',     action='store_true', help='Mosaic dominant tiles only')
args = parser.parse_args()

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
if not args.mosaic:
    print("Loading exp426 models...")
    with open(os.path.join(EXP426_DIR, 'feature_list.json')) as f:
        meta = json.load(f)
    feature_cols = meta['features']
    print(f"  Features: {len(feature_cols)}")

    models_fibric     = [pickle.load(open(os.path.join(EXP426_DIR, f'model_fibric_fold_{i}.pkl'), 'rb')) for i in range(N_FOLDS)]
    models_hemic_frac = [pickle.load(open(os.path.join(EXP426_DIR, f'model_hemic_frac_fold_{i}.pkl'), 'rb')) for i in range(N_FOLDS)]

    print("Loading exp410 models (for mask)...")
    with open(os.path.join(EXP410_DIR, 'feature_list.json')) as f:
        exp410_meta = json.load(f)
    exp410_feats  = exp410_meta['features']
    exp410_models = [pickle.load(open(os.path.join(EXP410_DIR, f'model_fold_{i}.pkl'), 'rb')) for i in range(N_FOLDS)]
    print("  All models loaded.")

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

def run_inference(geoms, region_name, out_dir, save_rgb=True):
    """Run exp426 inference for a given geometry. Returns dominant flat array."""
    t0 = time.time()

    with rasterio.open(DEM_PATH) as src:
        out, out_transform = rio_mask(src, geoms, crop=True, filled=True)
        ref_profile = src.profile.copy()
        H, W = out.shape[1], out.shape[2]
        ref_bounds = array_bounds(H, W, out_transform)

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
    print(f"  Valid pixels: {valid_mask.sum():,} / {n_pixels:,}")

    if valid_mask.sum() == 0:
        print(f"  No valid pixels — skipping")
        return None

    # exp410 mask
    peat_prob_flat = np.zeros(n_pixels)
    valid_idx = np.where(valid_mask)[0]
    for chunk_start in range(0, len(valid_idx), CHUNK_SIZE):
        chunk_idx = valid_idx[chunk_start:chunk_start+CHUNK_SIZE]
        chunk_data = {col: np.where(np.isnan(flat[col][chunk_idx]), 0, flat[col][chunk_idx])
                      if col in flat else np.zeros(len(chunk_idx))
                      for col in exp410_feats}
        X_chunk = pd.DataFrame(chunk_data)
        probs = np.stack([m.predict_proba(X_chunk)[:,1] for m in exp410_models]).mean(axis=0)
        peat_prob_flat[chunk_idx] = probs

    peat_mask = (peat_prob_flat >= PROB_THRESH) & valid_mask
    print(f"  Peat mask pixels: {peat_mask.sum():,}")

    # exp426 predictions
    peat_idx = np.where(peat_mask)[0]
    fibric_flat = np.full(n_pixels, np.nan)
    hemic_flat  = np.full(n_pixels, np.nan)
    sapric_flat = np.full(n_pixels, np.nan)

    for chunk_start in range(0, len(peat_idx), CHUNK_SIZE):
        chunk_idx = peat_idx[chunk_start:chunk_start+CHUNK_SIZE]
        chunk_data = {col: np.where(np.isnan(flat[col][chunk_idx]), 0, flat[col][chunk_idx])
                      if col in flat else np.zeros(len(chunk_idx))
                      for col in feature_cols}
        X_chunk = pd.DataFrame(chunk_data)

        # Step 1: Fibric
        pred_fibric = np.clip(
            np.stack([m.predict(X_chunk) for m in models_fibric]).mean(axis=0), 0, 100)
        # Step 2: Hemic fraction
        pred_hemic_frac = np.clip(
            np.stack([m.predict(X_chunk) for m in models_hemic_frac]).mean(axis=0), 0, 1)
        # Step 3: derive Hemic and Sapric
        remainder   = np.clip(100 - pred_fibric, 0, 100)
        pred_hemic  = remainder * pred_hemic_frac
        pred_sapric = remainder * (1 - pred_hemic_frac)

        fibric_flat[chunk_idx] = pred_fibric
        hemic_flat[chunk_idx]  = pred_hemic
        sapric_flat[chunk_idx] = pred_sapric

        if (chunk_start // CHUNK_SIZE) % 10 == 0:
            pct = 100*(chunk_start+len(chunk_idx))/len(peat_idx)
            print(f"    {pct:.0f}% complete...")

    # Verify sum
    check = fibric_flat[peat_mask] + hemic_flat[peat_mask] + sapric_flat[peat_mask]
    print(f"  Sum check: mean={check.mean():.4f}  min={check.min():.4f}  max={check.max():.4f}")

    # Dominant class
    dominant_flat = np.full(n_pixels, np.nan)
    stack = np.stack([fibric_flat, hemic_flat, sapric_flat], axis=0)
    dom_idx = np.argmax(stack, axis=0) + 1
    dominant_flat[peat_mask] = dom_idx[peat_mask].astype(float)
    dom_counts = {DOMINANT_LABELS[i]: int((dominant_flat[peat_mask]==i).sum()) for i in [1,2,3]}
    print(f"  Dominant counts: {dom_counts}")

    # Output profile
    out_profile = ref_profile.copy()
    out_profile.update({
        'height': H, 'width': W, 'transform': out_transform,
        'nodata': -9999.0, 'dtype': 'float32', 'count': 1, 'compress': 'lzw',
    })

    def save_raster(data_flat, fname):
        data_2d = data_flat.reshape(H, W).astype('float32')
        data_2d[np.isnan(data_2d)] = -9999.0
        out_path = os.path.join(out_dir, fname)
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            dst.write(data_2d, 1)
        valid = data_2d[data_2d != -9999.0]
        print(f"  Saved {fname}: mean={valid.mean():.1f}  min={valid.min():.1f}  max={valid.max():.1f}")

    save_raster(fibric_flat,   f'exp426_fibric_{region_name}.tif')
    save_raster(hemic_flat,    f'exp426_hemic_{region_name}.tif')
    save_raster(sapric_flat,   f'exp426_sapric_{region_name}.tif')
    save_raster(dominant_flat, f'exp426_dominant_{region_name}.tif')

    # RGB composite (scaled 0-255, R=Fibric G=Hemic B=Sapric)
    if save_rgb:
        rgb_profile = out_profile.copy()
        rgb_profile.update({'count': 3, 'dtype': 'uint8', 'nodata': 0})
        out_path = os.path.join(out_dir, f'exp426_rgb_{region_name}.tif')
        with rasterio.open(out_path, 'w', **rgb_profile) as dst:
            for band_i, data_flat in enumerate([fibric_flat, hemic_flat, sapric_flat], start=1):
                arr = data_flat.reshape(H, W)
                scaled = np.where(np.isnan(arr), 0,
                                  np.clip(arr / 100.0 * 255, 0, 255)).astype('uint8')
                dst.write(scaled, band_i)
        print(f"  Saved exp426_rgb_{region_name}.tif (R=Fibric G=Hemic B=Sapric)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")
    return dominant_flat, H, W, out_transform, ref_profile

# ── MOSAIC DOMINANT TILES ──────────────────────────────────────────────────────
if args.mosaic:
    import glob
    from rasterio.merge import merge
    print("Mosaicking dominant statewide tiles...")
    tiles = sorted(glob.glob(os.path.join(OUT_SW, 'exp426_dominant_tile_*.tif')))
    print(f"Found {len(tiles)} tiles")
    if len(tiles) == 0:
        print("No tiles found.")
        sys.exit(1)
    src_files = [rasterio.open(t) for t in tiles]
    mosaic, out_transform = merge(src_files, method='first', nodata=-9999.0)
    for src in src_files:
        src.close()
    with rasterio.open(tiles[0]) as src:
        out_profile = src.profile.copy()
    out_profile.update({
        'height': mosaic.shape[1], 'width': mosaic.shape[2],
        'transform': out_transform, 'nodata': -9999.0,
        'dtype': 'float32', 'count': 1, 'compress': 'lzw', 'bigtiff': 'YES',
    })
    out_path = os.path.join(OUT_SW, 'exp426_dominant_minnesota.tif')
    with rasterio.open(out_path, 'w', **out_profile) as dst:
        dst.write(mosaic[0], 1)
    valid = mosaic[0][mosaic[0] != -9999.0]
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"Mosaic saved: {out_path}")
    print(f"  Size: {size_gb:.2f} GB")
    print(f"  Dominant counts:")
    for i, label in DOMINANT_LABELS.items():
        print(f"    {label}: {(valid==i).sum():,} pixels ({100*(valid==i).mean():.1f}%)")
    sys.exit(0)

# ── SINGLE STATEWIDE TILE ──────────────────────────────────────────────────────
if args.tile_idx is not None:
    with fiona.open(GRID_SHP) as shp:
        tiles = list(shp)
    if args.tile_idx >= len(tiles):
        print(f"tile_idx {args.tile_idx} out of range")
        sys.exit(1)
    tile_feat = tiles[args.tile_idx]
    tile_id   = tile_feat['properties']['tile_id']
    out_path  = os.path.join(OUT_SW, f'exp426_dominant_{tile_id}.tif')
    if os.path.exists(out_path):
        print(f"Already exists — skipping: {out_path}")
        sys.exit(0)
    print(f"\nProcessing statewide tile: {tile_id}")
    geoms = [tile_feat['geometry']]
    result = run_inference(geoms, tile_id, OUT_SW, save_rgb=False)
    sys.exit(0)

# ── REGIONAL TILES (default) ───────────────────────────────────────────────────
if not args.statewide and args.tile_idx is None and not args.mosaic:
    for region_name, shp_path in REGIONAL_BOUNDARIES.items():
        print(f"\n{'='*60}")
        print(f"Region: {region_name}")
        print('='*60)
        with fiona.open(shp_path) as shp:
            geoms = [feat['geometry'] for feat in shp]
        run_inference(geoms, region_name, OUT_REG, save_rgb=True)
    print(f"\nAll regional tiles complete. Output: {OUT_REG}")
    print("RGB composite: use QGIS multiband renderer (Band 1=Red, Band 2=Green, Band 3=Blue)")
    print("Dominant: 1=Fibric(Oi)  2=Hemic(Oe)  3=Sapric(Oa)")
