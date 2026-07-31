"""
depth_spatial_inference.py
===========================
Spatial depth inference — predicts peat depth (cm) only on pixels
where exp410 peat probability >= 0.5.

Configurable: set DEPTH_EXP to any depth model exp_id.
Default: exp412

Boundaries: redlake, SE, SW
Output: 04_predictions/{DEPTH_EXP}/{DEPTH_EXP}_depth_{boundary}.tif
"""

import os, json, pickle, time
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.enums import Resampling
import fiona
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG — change DEPTH_EXP to swap model ───────────────────────
DEPTH_EXP   = 'exp419'   # <-- change to exp416 or exp417 after RFE
PROB_THRESH = 0.33
NODATA      = -9999.0
CHUNK_SIZE  = 500000

BASE    = '/scratch.global/ocon0444/peat_modeling'
COV_DIR = os.path.join(BASE, '00_data/covariates_10m')
BDY_DIR = os.path.join(BASE, '00_data/boundary')
MDL_DIR = os.path.join(BASE, '03_models')

BOUNDARIES = {
    'redlake': os.path.join(BDY_DIR, 'sample_boundary_RedLake.shp'),
    'SE':      os.path.join(BDY_DIR, 'sample_boundary_SE.shp'),
    'SW':      os.path.join(BDY_DIR, 'sample_boundary_SW.shp'),
}
PROB_RASTERS = {
    'redlake': os.path.join(BASE, '04_predictions/exp410/exp410_peat_prob_redlake.tif'),
    'SE':      os.path.join(BASE, '04_predictions/exp410/exp410_peat_prob_SE.tif'),
    'SW':      os.path.join(BASE, '04_predictions/exp410/exp410_peat_prob_SW.tif'),
}

# ── LOAD DEPTH MODEL ─────────────────────────────────────────────
model_dir = os.path.join(MDL_DIR, DEPTH_EXP)
print(f"Loading {DEPTH_EXP} models...")
with open(os.path.join(model_dir, 'feature_list.json')) as f:
    feat_data = json.load(f)
feature_cols = feat_data['features']
models = [pickle.load(open(os.path.join(model_dir, f'model_fold_{i}.pkl'), 'rb'))
          for i in range(5)]
print(f"  {len(models)} fold models, {len(feature_cols)} features")
print(f"  Filter: {feat_data.get('filter','depb>0 AND exp410>=0.5')}")

os.makedirs(os.path.join(BASE, f'04_predictions/{DEPTH_EXP}'), exist_ok=True)

# ── RASTER LOOKUP TABLES ─────────────────────────────────────────
S2_BAND_NAMES = ['B02','B03','B04','B05','B06','B07',
                 'B08','B8A','B11','B12','NDVI','SWDI']
S2_RASTERS = {
    's2_spring': 's2_spring_12bands.tif',
    's2_summer': 's2_summer_12bands.tif',
    's2_fall':   's2_fall_12bands.tif',
}
TC_BAND_NAMES = ['TCB','TCG','TCW']
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
SSURGO_RASTER = os.path.join(COV_DIR, 'MN_organic_soils_classified_FIXED_snapped.tif')
NWI_RASTER    = os.path.join(COV_DIR, 'mn_nwi_cowardin_10m.tif')

# ── HELPERS ──────────────────────────────────────────────────────
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

    # SSURGO
    ssurgo_needed = [c for c in feature_cols if c.startswith('MN_organic_soils_classified_')]
    if ssurgo_needed:
        raw = read_forced(SSURGO_RASTER, ref_bounds, H, W, ignore_nodata=True)
        for col in ssurgo_needed:
            try: clips[col] = (raw == int(col.split('_')[-1])).astype(float)
            except ValueError: clips[col] = np.zeros((H, W))

    # NWI (if in feature set)
    nwi_needed = [c for c in feature_cols if 'nwi' in c.lower()]
    if nwi_needed:
        raw_nwi = read_forced(NWI_RASTER, ref_bounds, H, W, ignore_nodata=True)
        if 'mn_nwi_cowardin_0' in feature_cols:
            clips['mn_nwi_cowardin_0'] = (raw_nwi == 0).astype(float)
        if 'mn_nwi_binary' in feature_cols or 'mn_nwi_merged_1_2' in feature_cols:
            merged = ((raw_nwi == 1) | (raw_nwi == 2)).astype(float)
            if 'mn_nwi_binary' in feature_cols:
                clips['mn_nwi_binary'] = merged
            if 'mn_nwi_merged_1_2' in feature_cols:
                clips['mn_nwi_merged_1_2'] = merged

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
        preds = np.stack([m.predict(chunk) for m in models], axis=0)
        result[idx[start:end]] = preds.mean(axis=0)
        print(f"    {end:,}/{len(idx):,} ({end/len(idx)*100:.0f}%)")
    return result

# ── MAIN INFERENCE LOOP ───────────────────────────────────────────
grand_start = time.time()
summary = {}

for bdy_name, shp_path in BOUNDARIES.items():
    print(f"\n{'#'*60}")
    print(f"BOUNDARY: {bdy_name}")
    print(f"{'#'*60}")

    geoms = load_geoms(shp_path)

    # Load exp410 peat probability mask
    with rasterio.open(PROB_RASTERS[bdy_name]) as src:
        prob_out, _ = rio_mask(src, geoms, crop=True, filled=True)
        prob_arr = prob_out[0].astype(float)
        nd = src.nodata if src.nodata else NODATA
        prob_arr[prob_arr == nd] = np.nan

    peat_mask_2d = (prob_arr >= PROB_THRESH) & (~np.isnan(prob_arr))
    pct = peat_mask_2d.sum() / peat_mask_2d.size * 100
    print(f"  Peat mask (>={PROB_THRESH}): {peat_mask_2d.sum():,} px ({pct:.1f}%)")

    print("  Clipping rasters...")
    clips, ref_profile, H, W = build_clips(geoms, feature_cols)
    peat_mask_flat = peat_mask_2d.flatten()

    t0 = time.time()
    X, valid_mask = stack_X(clips, feature_cols, H, W)
    predict_mask  = valid_mask & peat_mask_flat
    print(f"  Valid pixels   : {valid_mask.sum():,}")
    print(f"  Predict pixels : {predict_mask.sum():,} (peat mask applied)")

    depth_flat = predict_ensemble(X, predict_mask, models)
    depth_2d   = depth_flat.reshape(H, W)

    out_path = os.path.join(BASE, f'04_predictions/{DEPTH_EXP}',
                            f'{DEPTH_EXP}_depth_{bdy_name}.tif')
    with rasterio.open(out_path, 'w', **ref_profile) as dst:
        dst.write(depth_2d.astype('float32'), 1)

    valid_preds = depth_flat[depth_flat != NODATA]
    elapsed = (time.time() - t0) / 60
    print(f"  Saved  : {out_path}")
    if len(valid_preds):
        print(f"  Stats  : min={valid_preds.min():.1f}cm  mean={valid_preds.mean():.1f}cm  "
              f"max={valid_preds.max():.1f}cm  time={elapsed:.1f}min")

    summary[bdy_name] = {
        'exp_id': DEPTH_EXP, 'boundary': bdy_name,
        'peat_pixels': int(peat_mask_flat.sum()),
        'predict_pixels': int(predict_mask.sum()),
        'mean_depth_cm': round(float(valid_preds.mean()), 2) if len(valid_preds) else None,
        'max_depth_cm':  round(float(valid_preds.max()), 2) if len(valid_preds) else None,
        'elapsed_min': round(elapsed, 2),
    }

with open(os.path.join(BASE, f'04_predictions/{DEPTH_EXP}/{DEPTH_EXP}_depth_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

total = (time.time() - grand_start) / 60
print(f"\n{'='*60}")
print(f"ALL DONE in {total:.1f} min")
print(f"{'='*60}")
print(f"\n{'BOUNDARY':<12} {'MEAN_DEPTH':>12} {'MAX_DEPTH':>12} {'PIXELS':>12} {'MIN':>6}")
print('-'*58)
for bdy, s in summary.items():
    print(f"{bdy:<12} {str(s['mean_depth_cm'])+'cm':>12} "
          f"{str(s['max_depth_cm'])+'cm':>12} "
          f"{s['predict_pixels']:>12,} {s['elapsed_min']:>5.1f}")
