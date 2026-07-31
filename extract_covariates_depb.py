#!/usr/bin/env python3
"""
Extract covariate values for all points in depb_points_mn.csv.
Output: depb_features_extracted.csv
Same covariate set as binary_peat_features_extracted.csv.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

BASE_DIR  = "/scratch.global/ocon0444/peat_modeling"
COV_DIR   = f"{BASE_DIR}/00_data/covariates_10m"
INPUT_CSV = f"{BASE_DIR}/00_data/point_data/depb_points_mn.csv"
OUTPUT_CSV= f"{BASE_DIR}/00_data/processed/depb_features_extracted.csv"

# ── Raster definitions ────────────────────────────────────────────────────────

SINGLE_BAND = [
    "minnesota_dem_10m", "slope", "aspect", "hillshade",
    "planCurvature", "profileCurvature", "maximalCurvature",
    "breached_dem", "d8FlowAccumulation", "dInfFlowAccumulation",
    "devfrommeanelev_4m", "devfrommeanelev_8m", "devfrommeanelev_16m",
    "diffFromMeanElev",
    "relativeTopographicPosition_4m", "relativeTopographicPosition_8m",
    "relativeTopographicPosition_16m",
    "dist_to_water_10m", "dist_to_stream_10m", "dist_to_road_detailed_10m",
    "prism_ppt_mn", "prism_tmax_july_mn", "prism_tmean_mn", "prism_tmin_january_mn",
    "wetnessIndex",
    "gNATSGO_MN_26915",
    "npc_peatland_indicator_10m",
]

S2_BAND_NAMES = ["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12","NDVI","SWDI"]
MULTIBAND = {
    "s2_spring_12bands": ("s2_spring", S2_BAND_NAMES),
    "s2_summer_12bands": ("s2_summer", S2_BAND_NAMES),
    "s2_fall_12bands":   ("s2_fall",   S2_BAND_NAMES),
}

# Categorical rasters - one-hot encoded
# ignore_nodata=True for rasters where 0 is valid data
CATEGORICAL = [
    ("10m_quaternary_geology",          "quaternary_geology",           False),
    ("pennockLandformClass",            "pennockLandformClass",         False),
    ("geomorphons",                     "geomorphons",                  False),
    ("histosols_10m",                   "histosols_10m",                True),
    ("MN_organic_soils_classified_FIXED","MN_organic_soils_classified", True),
    ("MN_ANY_organic_component",        "MN_ANY_organic_component",     True),
    ("mn_nwi_cowardin_10m",             "mn_nwi_cowardin",              False),
]

# ── Load points ───────────────────────────────────────────────────────────────

print(f"Loading points from: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
print(f"  {len(df):,} points")

# Reproject lat/long (WGS84) to EPSG:5070 for raster sampling
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
x, y = transformer.transform(df['long'].values, df['lat'].values)
coords = list(zip(x, y))
print(f"  Reprojected to EPSG:5070")

result = df.copy()

# ── Single-band extraction ────────────────────────────────────────────────────

print(f"\nExtracting single-band rasters...")
for stem in SINGLE_BAND:
    fpath = f"{COV_DIR}/{stem}.tif"
    if not os.path.exists(fpath):
        print(f"  [SKIP] {stem} - not found")
        continue
    with rasterio.open(fpath) as src:
        nodata = src.nodata
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
        if nodata is not None:
            vals[vals == nodata] = np.nan
    result[stem] = vals
    nan_pct = np.isnan(vals).mean() * 100
    print(f"  + {stem} ({nan_pct:.1f}% NaN)")

# ── Multi-band S2 extraction ──────────────────────────────────────────────────

print(f"\nExtracting Sentinel-2 bands...")
for fname, (prefix, bands) in MULTIBAND.items():
    fpath = f"{COV_DIR}/{fname}.tif"
    if not os.path.exists(fpath):
        print(f"  [SKIP] {fname} - not found")
        continue
    with rasterio.open(fpath) as src:
        nodata = src.nodata
        sampled = list(src.sample(coords))
    for band_idx, band_name in enumerate(bands):
        col = f"{prefix}_{band_name}"
        vals = np.array([s[band_idx] for s in sampled], dtype=float)
        if nodata is not None:
            vals[vals == nodata] = np.nan
        result[col] = vals
    print(f"  + {fname} ({len(bands)} bands)")

# ── Categorical one-hot extraction ────────────────────────────────────────────

print(f"\nExtracting categorical rasters (one-hot)...")
for fname, prefix, ignore_nodata in CATEGORICAL:
    fpath = f"{COV_DIR}/{fname}.tif"
    if not os.path.exists(fpath):
        print(f"  [SKIP] {fname} - not found")
        continue
    with rasterio.open(fpath) as src:
        nodata = src.nodata
        vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
        if nodata is not None and not ignore_nodata:
            vals[vals == nodata] = np.nan
    raw = pd.Series(vals, name=f"{prefix}_raw")
    dummies = pd.get_dummies(raw.astype("Int64"), prefix=prefix)
    for col in dummies.columns:
        result[col] = dummies[col].values
    nan_pct = np.isnan(vals).mean() * 100
    print(f"  + {fname} -> {list(dummies.columns)} ({nan_pct:.1f}% NaN)")

# ── Save ──────────────────────────────────────────────────────────────────────

print(f"\nFinal shape: {result.shape}")
print(f"Saving to: {OUTPUT_CSV}")
result.to_csv(OUTPUT_CSV, index=False)

print("\nNaN summary (feature columns only):")
meta = ['lat', 'long', 'depb']
feat_cols = [c for c in result.columns if c not in meta]
nan_counts = result[feat_cols].isna().sum()
nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)
if len(nan_cols) > 0:
    for col, cnt in nan_cols.items():
        print(f"  {col}: {cnt:,} ({cnt/len(result)*100:.1f}%)")
else:
    print("  No NaNs in feature columns!")

print("\nExtraction complete!")
