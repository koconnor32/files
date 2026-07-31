#!/usr/bin/env python3
"""
extract_covariates_carbon_bd.py
Extracts covariates for carbon and bulk density point sets.
Runs both in one job to save queue time.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

COV_DIR = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m/"

POINT_SETS = [
    {
        'input':  '/scratch.global/ocon0444/peat_modeling/00_data/point_data/carbon_points.csv',
        'output': '/scratch.global/ocon0444/peat_modeling/00_data/processed/carbon_features_extracted.csv',
        'label':  'carbon',
    },
    {
        'input':  '/scratch.global/ocon0444/peat_modeling/00_data/point_data/bd_points.csv',
        'output': '/scratch.global/ocon0444/peat_modeling/00_data/processed/bd_features_extracted.csv',
        'label':  'bulk_density',
    },
]

SINGLE_BAND = [
    "minnesota_dem_10m.tif", "slope.tif", "aspect.tif", "hillshade.tif",
    "planCurvature.tif", "profileCurvature.tif", "maximalCurvature.tif",
    "breached_dem.tif", "d8FlowAccumulation.tif", "dInfFlowAccumulation.tif",
    "wetnessIndex.tif", "devfrommeanelev_4m.tif", "devfrommeanelev_8m.tif",
    "devfrommeanelev_16m.tif", "diffFromMeanElev.tif",
    "relativeTopographicPosition_4m.tif", "relativeTopographicPosition_8m.tif",
    "relativeTopographicPosition_16m.tif", "dist_to_water_10m.tif",
    "dist_to_stream_10m.tif", "dist_to_road_detailed_10m.tif",
    "prism_ppt_mn.tif", "prism_tmax_july_mn.tif", "prism_tmean_mn.tif",
    "prism_tmin_january_mn.tif", "mn_nwi_cowardin_10m.tif",
    "histosols_10m_snapped.tif", "MN_ANY_organic_component_snapped.tif",
    "MN_organic_soils_classified_FIXED_snapped.tif",
    "gNATSGO_MN_26915.tif", "npc_peatland_indicator_10m.tif",
]
SENTINEL2_BANDS = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","NDVI","SWDI"]
TC_BANDS        = ["TCB","TCG","TCW"]
MULTIBAND = [
    ("s2_spring_12bands.tif",     "s2_spring",   SENTINEL2_BANDS),
    ("s2_summer_12bands.tif",     "s2_summer",   SENTINEL2_BANDS),
    ("s2_fall_12bands.tif",       "s2_fall",     SENTINEL2_BANDS),
    ("tc_spring_merged_5070.tif", "tc_spring",   TC_BANDS),
    ("tc_summer_merged_5070.tif", "tc_summer",   TC_BANDS),
    ("tc_fall_merged_5070.tif",   "tc_fall",     TC_BANDS),
]
CATEGORICAL = [
    "10m_quaternary_geology.tif",
    "pennockLandformClass.tif",
    "geomorphons.tif",
]

def sample_raster(path, xs, ys, band=1):
    with rasterio.open(path) as src:
        vals = np.array([v[0] for v in src.sample(zip(xs, ys), indexes=band)], dtype=float)
        nd = src.nodata
        if nd is not None:
            vals[vals == nd] = np.nan
    return vals

transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)

for ps in POINT_SETS:
    print(f"\n{'='*60}")
    print(f"Processing: {ps['label']}")
    print('='*60)

    df = pd.read_csv(ps['input'])
    print(f"  {len(df)} points loaded")

    xs, ys = transformer.transform(df['lon'].values, df['lat'].values)

    print("  Extracting single-band rasters...")
    for fname in SINGLE_BAND:
        path = os.path.join(COV_DIR, fname)
        col  = fname.replace('.tif', '')
        if not os.path.exists(path):
            df[col] = np.nan
            continue
        df[col] = sample_raster(path, xs, ys)

    print("  Extracting multi-band rasters...")
    for fname, prefix, band_names in MULTIBAND:
        path = os.path.join(COV_DIR, fname)
        if not os.path.exists(path):
            for b in band_names:
                df[f"{prefix}_{b}"] = np.nan
            continue
        for i, bname in enumerate(band_names, start=1):
            df[f"{prefix}_{bname}"] = sample_raster(path, xs, ys, band=i)

    print("  Extracting categorical rasters...")
    for fname in CATEGORICAL:
        path = os.path.join(COV_DIR, fname)
        prefix = fname.replace('.tif', '')
        if not os.path.exists(path):
            continue
        raw  = sample_raster(path, xs, ys)
        cats = [int(v) for v in np.unique(raw[~np.isnan(raw)]) if v != 0]
        for c in cats:
            df[f"{prefix}_{c}"] = (raw == c).astype(int)

    # Summary
    covar_cols = [c for c in df.columns if c not in
                  ['point_id','source','DNR Peat Inventory ID #','Depthnum',
                   'Top Depth (cm)','Bottom Depth (cm)','mid_depth_cm',
                   'Classification','Degree of Decomposition',
                   'carbon_pct','carbon_source','bulk_density_gcc','lat','lon']]
    nan_counts = pd.Series({c: df[c].isna().sum() for c in covar_cols})
    high_nan   = nan_counts[nan_counts > len(df) * 0.10]
    print(f"  Total columns: {len(df.columns)}")
    if len(high_nan) > 0:
        print(f"  Covariates >10% NaN: {len(high_nan)}")
        for col, n in high_nan.items():
            print(f"    {col}: {100*n/len(df):.1f}%")
    else:
        print("  All covariates <10% NaN")

    df.to_csv(ps['output'], index=False)
    print(f"  Saved {len(df):,} rows x {len(df.columns)} cols → {ps['output']}")

print("\nDone.")
