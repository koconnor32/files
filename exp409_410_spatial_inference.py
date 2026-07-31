"""
exp409_410_spatial_inference.py
================================
Spatial peat probability inference for exp409 and exp410.
Models already trained — loads from 03_models/exp409 and exp410.

exp409 — continuous + TC only, no categoricals
exp410 — continuous + TC + NWI binary (0 vs 1+2), no other categoricals

Output naming:
  04_predictions/exp409/exp409_peat_prob_{boundary}.tif
  04_predictions/exp410/exp410_peat_prob_{boundary}.tif
"""

import os, json, pickle, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
import fiona
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────
BASE    = '/scratch.global/ocon0444/peat_modeling'
COV_DIR = os.path.join(BASE, '00_data/covariates_10m')
BDY_DIR = os.path.join(BASE, '00_data/boundary')
MDL_DIR = os.path.join(BASE, '03_models')

CHUNK_SIZE = 500000
NODATA     = -9999.0

BOUNDARIES = {
    'greenwood': os.path.join(BDY_DIR, 'greenwood_area_5070.shp'),
    'redlake':   os.path.join(BDY_DIR, 'sample_boundary_RedLake.shp'),
    'SE':        os.path.join(BDY_DIR, 'sample_boundary_SE.shp'),
    'SW':        os.path.join(BDY_DIR, 'sample_boundary_SW.shp'),
}

EXPERIMENTS = ['exp409', 'exp410']

S2_BAND_NAMES = ['B02','B03','B04','B05','B06','B07',
                 'B08','B8A','B11','B12','NDVI','SWDI']
S2_RASTERS = {
    's2_spring': 's2_spring_12bands.tif',
    's2_summer': 's2_summer_12bands.tif',
    's2_fall':   's2_fall_12bands.tif',
}
TC_BAND_NAMES = ['TCB', 'TCG', 'TCW']
TC_RASTERS = {
    'tc_spring': 'tc_spring_merged_5070.tif',
    'tc_summer': 'tc_summer_merged_5070.tif',
    'tc_fall':   'tc_fall_merged_5070.tif',
}
SINGLE_BAND_RASTERS = {
    'minnesota_dem_10m':               'minnesota_dem_10m.tif',
    'slope':                           'slope.tif',
    'aspect':                          'aspect.tif',
    'hillshade':                       'hillshade.tif',
    'planCurvature':                   'planCurvature.tif',
    'profileCurvature':                'profileCurvature.tif',
    'maximalCurvature':                'maximalCurvature.tif',
    'breached_dem':                    'breached_dem.tif',
    'd8FlowAccumulation':              'd8FlowAccumulation.tif',
    'dInfFlowAccumulation':            'dInfFlowAccumulation.tif',
    'diffFromMeanElev':                'diffFromMeanElev.tif',
    'devfrommeanelev_4m':              'devfrommeanelev_4m.tif',
    'devfrommeanelev_8m':              'devfrommeanelev_8m.tif',
    'devfrommeanelev_16m':             'devfrommeanelev_16m.tif',
    'relativeTopographicPosition_4m':  'relativeTopographicPosition_4m.tif',
    'relativeTopographicPosition_8m':  'relativeTopographicPosition_8m.tif',
    'relativeTopographicPosition_16m': 'relativeTopographicPosition_16m.tif',
    'dist_to_water_10m':               'dist_to_water_10m.tif',
    'dist_to_stream_10m':              'dist_to_stream_10m.tif',
    'dist_to_road_detailed_10m':       'dist_to_road_detailed_10m.tif',
    'prism_ppt_mn':                    'prism_ppt_mn.tif',
    'prism_tmax_july_mn':              'prism_tmax_july_mn.tif',
    'prism_tmean_mn':                  'prism_tmean_mn.tif',
    'prism_tmin_january_mn':           'prism_tmin_january_mn.tif',
    'wetnessIndex':                    'wetnessIndex.tif',
}
NWI_RASTER = os.path.join(COV_DIR, 'mn_nwi_cowardin_10m.tif')

# ── LOAD MODELS ───────────────────────────────────────────────────
print("Loading models and feature lists...")
exp_data = {}
for exp_id in EXPERIMENTS:
    model_dir = os.path.join(MDL_DIR, exp_id)
    models = []
    for i in range(5):
        with open(os.path.join(model_dir, f'model_fold_{i}.pkl'), 'rb') as f:
            models.append(pickle.load(f))
    with open(os.path.join(model_dir, 'feature_list.json')) as f:
        feat_data = json.load(f)
    exp_data[exp_id] = {
        'models': models,
        'feature_cols': feat_data['features'],
        'nwi_type': feat_data.get('nwi_type', 'unknown'),
    }
    os.makedirs(os.path.join(BASE, f'04_predictions/{exp_id}'), exist_ok=True)
    print(f"  {exp_id}: {len(models)} models, "
          f"{len(feat_data['features'])} features, "
          f"NWI={feat_data.get('nwi_type','unknown')}")
print()

# ── HELPERS ───────────────────────────────────────────────────────
def load_geoms(shp_path):
    return [shape(f['geometry']) for f in fiona.open(shp_path)]

def read_forced(path, ref_bounds, H, W, band=1, ignore_nodata=False):
    with rasterio.open(path) as src:
        window = src.window(*ref_bounds)
        data = src.read(band, window=window, out_shape=(H, W),
                        resampling=Resampling.bilinear).astype(float)
        nd = src.nodata
        if nd is not None and not ignore_nodata:
            data[data == nd] = np.nan
    return data

def build_clips(geoms, feature_cols):
    dem_path = os.path.join(COV_DIR, 'minnesota_dem_10m.tif')
    with rasterio.open(dem_path) as src:
        out, out_transform = rio_mask(src, geoms, crop=True, filled=True)
        H, W = out.shape[1], out.shape[2]
        ref_profile = src.profile.copy()
        ref_profile.update({
            'height': H, 'width': W, 'transform': out_transform,
            'nodata': NODATA, 'dtype': 'float32', 'count': 1, 'compress': 'lzw',
        })
        arr = out[0].astype(float)
        if src.nodata: arr[arr == src.nodata] = np.nan
        ref_bounds = rasterio.transform.array_bounds(H, W, out_transform)

    clips = {'minnesota_dem_10m': arr}
    print(f"  Grid: {H} x {W} = {H*W:,} pixels")

    for col, fname in SINGLE_BAND_RASTERS.items():
        if col not in feature_cols or col == 'minnesota_dem_10m': continue
        clips[col] = read_forced(os.path.join(COV_DIR, fname), ref_bounds, H, W)

    for prefix, fname in S2_RASTERS.items():
        for i, band in enumerate(S2_BAND_NAMES, 1):
            col = f'{prefix}_{band}'
            if col not in feature_cols: continue
            clips[col] = read_forced(os.path.join(COV_DIR, fname), ref_bounds, H, W, band=i)

    for prefix, fname in TC_RASTERS.items():
        for i, band in enumerate(TC_BAND_NAMES, 1):
            col = f'{prefix}_{band}'
            if col not in feature_cols: continue
            clips[col] = read_forced(os.path.join(COV_DIR, fname), ref_bounds, H, W, band=i)

    # NWI — exp410 uses mn_nwi_cowardin_0 and mn_nwi_binary
    nwi_needed = [c for c in feature_cols if 'nwi' in c.lower()]
    if nwi_needed:
        raw_nwi = read_forced(NWI_RASTER, ref_bounds, H, W, ignore_nodata=True)
        if 'mn_nwi_cowardin_0' in feature_cols:
            clips['mn_nwi_cowardin_0'] = (raw_nwi == 0).astype(float)
        if 'mn_nwi_binary' in feature_cols:
            clips['mn_nwi_binary'] = ((raw_nwi == 1) | (raw_nwi == 2)).astype(float)
        # fallback for individual cols if stored that way
        for v in [1, 2]:
            col = f'mn_nwi_cowardin_{v}'
            if col in feature_cols:
                clips[col] = (raw_nwi == v).astype(float)

    return clips, ref_profile, H, W

def stack_X(clips, feature_cols, H, W):
    arrays = []
    for col in feature_cols:
        if col in clips:
            arrays.append(clips[col].flatten())
        else:
            print(f"  WARNING: {col} missing — filling NaN")
            arrays.append(np.full(H * W, np.nan))
    X = np.column_stack(arrays)
    return X, ~np.any(np.isnan(X), axis=1)

def predict_ensemble(X, mask, models):
    result = np.full(X.shape[0], NODATA)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        print("  WARNING: no valid pixels"); return result
    for start in range(0, len(idx), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(idx))
        chunk = np.where(np.isnan(X[idx[start:end]]), 0, X[idx[start:end]])
        fold_probs = np.stack([m.predict_proba(chunk)[:, 1] for m in models], axis=0)
        result[idx[start:end]] = fold_probs.mean(axis=0)
        print(f"    {end:,}/{len(idx):,} ({end/len(idx)*100:.0f}%)")
    return result

# ── MAIN LOOP ─────────────────────────────────────────────────────
grand_start = time.time()
summary = {}

all_feature_cols = set()
for edata in exp_data.values():
    all_feature_cols.update(edata['feature_cols'])

for bdy_name, shp_path in BOUNDARIES.items():
    print(f"\n{'#'*60}")
    print(f"BOUNDARY: {bdy_name}")
    print(f"{'#'*60}")

    geoms = load_geoms(shp_path)
    print("  Clipping rasters (once for both experiments)...")
    clips, ref_profile, H, W = build_clips(geoms, list(all_feature_cols))

    for exp_id, edata in exp_data.items():
        t0 = time.time()
        print(f"\n  --- {exp_id} | NWI={edata['nwi_type']} ---")

        X, valid_mask = stack_X(clips, edata['feature_cols'], H, W)
        print(f"  Valid pixels: {valid_mask.sum():,} / {H*W:,}")

        prob_flat = predict_ensemble(X, valid_mask, edata['models'])
        prob_2d   = prob_flat.reshape(H, W)

        out_path = os.path.join(BASE, f'04_predictions/{exp_id}',
                                f'{exp_id}_peat_prob_{bdy_name}.tif')
        with rasterio.open(out_path, 'w', **ref_profile) as dst:
            dst.write(prob_2d.astype('float32'), 1)

        valid_probs = prob_flat[prob_flat != NODATA]
        pct_peat    = (valid_probs >= 0.5).sum() / len(valid_probs) * 100
        elapsed     = (time.time() - t0) / 60

        print(f"  Saved : {out_path}")
        print(f"  Stats : mean={valid_probs.mean():.3f}  "
              f"pct>=0.5={pct_peat:.1f}%  time={elapsed:.1f}min")

        summary[f'{exp_id}_{bdy_name}'] = {
            'exp_id': exp_id, 'boundary': bdy_name,
            'nwi_type': edata['nwi_type'],
            'valid_pixels': int(valid_mask.sum()),
            'pct_peat': round(pct_peat, 2),
            'mean_prob': round(float(valid_probs.mean()), 4),
            'elapsed_min': round(elapsed, 2),
        }

with open(os.path.join(BASE, '04_predictions/exp409_410_inference_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

total_min = (time.time() - grand_start) / 60
print(f"\n{'='*60}")
print(f"ALL DONE in {total_min:.1f} min")
print(f"{'='*60}")
print(f"\n{'KEY':<32} {'MEAN_PROB':>10} {'PCT_PEAT':>10} {'MIN':>6}")
print('-'*62)
for key, s in summary.items():
    print(f"{key:<32} {s['mean_prob']:>10.3f} "
          f"{s['pct_peat']:>9.1f}% {s['elapsed_min']:>5.1f}")
