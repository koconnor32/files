#!/bin/bash
#SBATCH --job-name=gam_prob_rl
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=2:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/gam_prob%j%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/gam_prob%j%j.err

echo "=============================="
echo "gam prob — SE"
echo "Started: $(date)"
echo "=============================="

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os, json, pickle
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import fiona
from shapely.geometry import shape
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')

BASE_M   = '/scratch.global/ocon0444/peat_modeling'
COV_DIR  = f'{BASE_M}/00_data/covariates_10m'
PRED_DIR = f'{BASE_M}/04_predictions'
MDL_DIR  = f'{BASE_M}/03_models'
CRS      = rasterio.crs.CRS.from_epsg(5070)
PROB_THRESH = 0.33
EXP410   = f'{PRED_DIR}/exp410_statewide/exp410_prob_minnesota_regrid.tif'
SHP_PATH = f'{BASE_M}/00_data/boundary/sample_boundary_SE.shp'

MODEL_NAME  = 'gam'
TARGET_NAME = 'prob'
IS_CLASSIFIER = TARGET_NAME == 'prob'
IS_GAM = MODEL_NAME == 'gam'

if TARGET_NAME == 'prob':
    PKL  = f'{MDL_DIR}/exp_{MODEL_NAME}_peat/{MODEL_NAME}_peat_prob_final.pkl'
    FEAT = f'{MDL_DIR}/exp_{MODEL_NAME}_peat/{MODEL_NAME}_peat_prob_features.json'
    OUT_DIR = f'{PRED_DIR}/exp_{MODEL_NAME}_peat'
else:
    PKL  = f'{MDL_DIR}/exp_{MODEL_NAME}_depth/{MODEL_NAME}_depth_final.pkl'
    FEAT = f'{MDL_DIR}/exp_{MODEL_NAME}_depth/{MODEL_NAME}_depth_features.json'
    OUT_DIR = f'{PRED_DIR}/exp_{MODEL_NAME}_depth'

os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = f'{OUT_DIR}/{MODEL_NAME}_{TARGET_NAME}_SE.tif'

if os.path.exists(OUT_PATH):
    print('Already exists: ' + OUT_PATH)
    exit(0)

BAND_MAP = {
    's2_spring_B02':1,'s2_spring_B03':2,'s2_spring_B04':3,'s2_spring_B05':4,
    's2_spring_B06':5,'s2_spring_B07':6,'s2_spring_B08':7,'s2_spring_B8A':8,
    's2_spring_B11':9,'s2_spring_B12':10,'s2_spring_NDVI':11,'s2_spring_SWDI':12,
    's2_summer_B02':1,'s2_summer_B03':2,'s2_summer_B04':3,'s2_summer_B05':4,
    's2_summer_B06':5,'s2_summer_B07':6,'s2_summer_B08':7,'s2_summer_B8A':8,
    's2_summer_B11':9,'s2_summer_B12':10,'s2_summer_NDVI':11,'s2_summer_SWDI':12,
    's2_fall_B02':1,'s2_fall_B03':2,'s2_fall_B04':3,'s2_fall_B05':4,
    's2_fall_B06':5,'s2_fall_B07':6,'s2_fall_B08':7,'s2_fall_B8A':8,
    's2_fall_B11':9,'s2_fall_B12':10,'s2_fall_NDVI':11,'s2_fall_SWDI':12,
    'tc_spring_TCB':1,'tc_spring_TCG':2,'tc_spring_TCW':3,
    'tc_summer_TCB':1,'tc_summer_TCG':2,'tc_summer_TCW':3,
    'tc_fall_TCB':1,'tc_fall_TCG':2,'tc_fall_TCW':3,
}
RASTER_MAP = {
    's2_spring': f'{COV_DIR}/s2_spring_12bands.tif',
    's2_summer': f'{COV_DIR}/s2_summer_12bands.tif',
    's2_fall':   f'{COV_DIR}/s2_fall_12bands.tif',
    'tc_spring': f'{COV_DIR}/tc_spring_merged_5070.tif',
    'tc_summer': f'{COV_DIR}/tc_summer_merged_5070.tif',
    'tc_fall':   f'{COV_DIR}/tc_fall_merged_5070.tif',
}
SINGLE_MAP = {
    'minnesota_dem_10m':               f'{COV_DIR}/minnesota_dem_10m.tif',
    'relativeTopographicPosition_4m':  f'{COV_DIR}/relativeTopographicPosition_4m.tif',
    'relativeTopographicPosition_8m':  f'{COV_DIR}/relativeTopographicPosition_8m.tif',
    'relativeTopographicPosition_16m': f'{COV_DIR}/relativeTopographicPosition_16m.tif',
    'dist_to_water_10m':               f'{COV_DIR}/dist_to_water_10m.tif',
    'dist_to_stream_10m':              f'{COV_DIR}/dist_to_stream_10m.tif',
    'dist_to_road_detailed_10m':       f'{COV_DIR}/dist_to_road_detailed_10m.tif',
    'dist_from_waterbody_edge_10m':    f'{COV_DIR}/dist_from_waterbody_edge_10m.tif',
    'prism_ppt_mn':                    f'{COV_DIR}/prism_ppt_mn.tif',
    'prism_tmax_july_mn':              f'{COV_DIR}/prism_tmax_july_mn.tif',
    'prism_tmean_mn':                  f'{COV_DIR}/prism_tmean_mn.tif',
    'prism_tmin_january_mn':           f'{COV_DIR}/prism_tmin_january_mn.tif',
    'slope':                           f'{COV_DIR}/slope.tif',
    'aspect':                          f'{COV_DIR}/aspect.tif',
    'wetnessIndex':                    f'{COV_DIR}/wetnessIndex.tif',
    'diffFromMeanElev':                f'{COV_DIR}/diffFromMeanElev.tif',
    'devfrommeanelev_4m':              f'{COV_DIR}/devfrommeanelev_4m.tif',
    'devfrommeanelev_8m':              f'{COV_DIR}/devfrommeanelev_8m.tif',
    'devfrommeanelev_16m':             f'{COV_DIR}/devfrommeanelev_16m.tif',
    'planCurvature':                   f'{COV_DIR}/planCurvature.tif',
    'profileCurvature':                f'{COV_DIR}/profileCurvature.tif',
    'maximalCurvature':                f'{COV_DIR}/maximalCurvature.tif',
    'geomorphons':                     f'{COV_DIR}/geomorphons.tif',
    'mn_nwi_binary':                   f'{COV_DIR}/mn_nwi_cowardin_10m.tif',
}

def read_feat(feat, geom_json):
    if feat == 'mn_nwi_binary':
        with rasterio.open(SINGLE_MAP['mn_nwi_binary']) as src:
            out, tf = rio_mask(src, geom_json, crop=True)
            raw = out.squeeze().astype(np.float32)
        return ((raw==1)|(raw==2)).astype(np.float32), tf
    for prefix in ['s2_spring','s2_summer','s2_fall','tc_spring','tc_summer','tc_fall']:
        if feat.startswith(prefix) and feat in BAND_MAP:
            with rasterio.open(RASTER_MAP[prefix]) as src:
                out, tf = rio_mask(src, geom_json, crop=True, indexes=BAND_MAP[feat])
                data = out.astype(np.float32)
                if src.nodata: data[data==src.nodata] = np.nan
            return data.squeeze(), tf
    if feat in SINGLE_MAP:
        with rasterio.open(SINGLE_MAP[feat]) as src:
            out, tf = rio_mask(src, geom_json, crop=True)
            data = out.squeeze().astype(np.float32)
            if src.nodata: data[data==src.nodata] = np.nan
        if feat == 'dist_from_waterbody_edge_10m':
            data = np.where(np.isnan(data), 99999, data)
        return data, tf
    return None, None

# Load model
print('Loading ' + MODEL_NAME + ' ' + TARGET_NAME + ' model...')
with open(PKL, 'rb') as f:
    obj = pickle.load(f)
if IS_GAM and isinstance(obj, dict):
    model  = obj['model']
    scaler = obj['scaler']
else:
    model  = obj
    scaler = None
with open(FEAT) as f:
    feature_cols = json.load(f)['features']
print('Features: ' + str(len(feature_cols)))

# Load region
with fiona.open(SHP_PATH) as src:
    geoms = [shape(f['geometry']) for f in src]
region_geom = unary_union(geoms)
geom_json   = [region_geom.__geo_interface__]

# Reference grid
with rasterio.open(SINGLE_MAP['minnesota_dem_10m']) as src:
    ref_out, ref_tf = rio_mask(src, geom_json, crop=True)
h, w = ref_out.squeeze().shape
print('Grid: ' + str(h) + ' x ' + str(w))

# Peat mask
with rasterio.open(EXP410) as src:
    out, _ = rio_mask(src, geom_json, crop=True)
    prob   = out.squeeze().astype(np.float32)
    if src.nodata: prob[prob==src.nodata] = 0
peat_mask = (prob >= PROB_THRESH).flatten()
print('Peat pixels: ' + str(peat_mask.sum()))

# Read features
stack = np.full((len(feature_cols), h, w), np.nan, dtype=np.float32)
for fi, feat in enumerate(feature_cols):
    arr, _ = read_feat(feat, geom_json)
    if arr is not None and arr.shape == (h, w):
        stack[fi] = arr

px = stack.reshape(len(feature_cols), -1).T
# Fill NaN with median
for fi in range(px.shape[1]):
    col = px[:, fi]
    nan_mask = np.isnan(col)
    if nan_mask.any():
        med = np.nanmedian(col)
        px[nan_mask, fi] = med if not np.isnan(med) else 0

valid = peat_mask.copy()
out_v = np.full(h*w, -9999, dtype=np.float32)

if valid.sum() > 0:
    X = px[valid]
    if IS_GAM and scaler is not None:
        X = scaler.transform(X)
    if IS_CLASSIFIER:
        out_v[valid] = model.predict_proba(X)[:, 1].astype(np.float32)
    else:
        out_v[valid] = model.predict(X).astype(np.float32)

v = out_v[out_v != -9999]
print('Valid pixels: ' + str(len(v)))
print('Mean: ' + str(round(float(v.mean()), 4)))
print('Range: [' + str(round(float(v.min()), 3)) + ', ' + str(round(float(v.max()), 3)) + ']')

prof = {'driver':'GTiff','dtype':'float32','count':1,
        'height':h,'width':w,'crs':CRS,
        'transform':ref_tf,'nodata':-9999,'compress':'lzw'}
with rasterio.open(OUT_PATH, 'w', **prof) as dst:
    dst.write(out_v.reshape(h, w), 1)
print('Saved: ' + OUT_PATH)
ENDPYTHON

echo ""
echo "=============================="
echo "Finished: $(date)"
echo "=============================="
