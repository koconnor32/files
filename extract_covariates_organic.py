#!/usr/bin/env python3
"""
Covariate extraction for organic composition / hydraulic conductivity modeling.
Extracts values from all rasters in covariates_10m/ at ~18K organic point locations.
Handles: single-band, multi-band, and categorical rasters.

Usage:
    python extract_covariates_organic.py
"""
import os
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
POINTS_CSV = "/scratch.global/ocon0444/peat_modeling/00_data/point_data/organic_composition_0_50cm_combined.csv"
COV_DIR    = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m/"
OUTPUT_CSV = "/scratch.global/ocon0444/peat_modeling/00_data/processed/organic_composition_features_extracted.csv"

# ─────────────────────────────────────────────
# RASTER DEFINITIONS
# ─────────────────────────────────────────────
# Single-band continuous rasters → column name = filename stem
SINGLE_BAND = [
    "minnesota_dem_10m.tif",
    "slope.tif",
    "aspect.tif",
    "hillshade.tif",
    "planCurvature.tif",
    "profileCurvature.tif",
    "maximalCurvature.tif",
    "breached_dem.tif",
    "d8FlowAccumulation.tif",
    "dInfFlowAccumulation.tif",
    "wetnessIndex.tif",
    "devfrommeanelev_4m.tif",
    "devfrommeanelev_8m.tif",
    "devfrommeanelev_16m.tif",
    "diffFromMeanElev.tif",
    "relativeTopographicPosition_4m.tif",
    "relativeTopographicPosition_8m.tif",
    "relativeTopographicPosition_16m.tif",
    "dist_to_water_10m.tif",
    "dist_to_stream_10m.tif",
    "dist_to_road_detailed_10m.tif",
    "dist_from_waterbody_edge_10m.tif",
    "prism_ppt_mn.tif",
    "prism_tmax_july_mn.tif",
    "prism_tmean_mn.tif",
    "prism_tmin_january_mn.tif",
    "mn_nwi_cowardin_10m.tif",
    "histosols_10m_snapped.tif",
    "MN_ANY_organic_component_snapped.tif",
    "MN_organic_soils_classified_FIXED_snapped.tif",
    "gNATSGO_MN_26915.tif",
    "npc_peatland_indicator_10m.tif",
]

# Categorical rasters → one-hot encoded
CATEGORICAL = [
    "10m_quaternary_geology.tif",
    "pennockLandformClass.tif",
    "geomorphons.tif",
]

# Multi-band rasters → (filename, band_prefix, band_names)
SENTINEL2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "SWDI"]
TC_BANDS        = ["TCB", "TCG", "TCW"]
MULTIBAND = [
    ("s2_spring_12bands.tif",   "s2_spring",   SENTINEL2_BANDS),
    ("s2_summer_12bands.tif",   "s2_summer",   SENTINEL2_BANDS),
    ("s2_fall_12bands.tif",     "s2_fall",     SENTINEL2_BANDS),
    ("tc_spring_merged_5070.tif", "tc_spring", TC_BANDS),
    ("tc_summer_merged_5070.tif", "tc_summer", TC_BANDS),
    ("tc_fall_merged_5070.tif",   "tc_fall",   TC_BANDS),
]

# ─────────────────────────────────────────────
# LOAD POINTS
# ─────────────────────────────────────────────
print("Loading points...")
df = pd.read_csv(POINTS_CSV)
print(f"  {len(df)} points loaded")

# Transform lat/lon (EPSG:4326) → EPSG:5070 for raster extraction
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
xs, ys = transformer.transform(df['lon'].values, df['lat'].values)
print(f"  Coordinates transformed to EPSG:5070")

# ─────────────────────────────────────────────
# HELPER: sample raster at point locations
# ─────────────────────────────────────────────
def sample_raster(path, xs, ys, band=1):
    """Sample a single band raster at (x, y) coordinates. Returns array of values."""
    with rasterio.open(path) as src:
        coords = list(zip(xs, ys))
        vals = np.array([v[0] for v in src.sample(coords, indexes=band)], dtype=float)
        nd = src.nodata
        if nd is not None:
            vals[vals == nd] = np.nan
    return vals

# ─────────────────────────────────────────────
# EXTRACT SINGLE-BAND RASTERS
# ─────────────────────────────────────────────
print("\nExtracting single-band rasters...")
for fname in SINGLE_BAND:
    path = os.path.join(COV_DIR, fname)
    col  = fname.replace('.tif', '')
    if not os.path.exists(path):
        print(f"  WARNING: {fname} not found, skipping")
        df[col] = np.nan
        continue
    df[col] = sample_raster(path, xs, ys)
    n_valid = df[col].notna().sum()
    print(f"  {col}: {n_valid}/{len(df)} valid ({100*n_valid/len(df):.1f}%)")

# ─────────────────────────────────────────────
# EXTRACT MULTI-BAND RASTERS
# ─────────────────────────────────────────────
print("\nExtracting multi-band rasters...")
for fname, prefix, band_names in MULTIBAND:
    path = os.path.join(COV_DIR, fname)
    if not os.path.exists(path):
        print(f"  WARNING: {fname} not found, skipping all bands")
        for b in band_names:
            df[f"{prefix}_{b}"] = np.nan
        continue
    for i, bname in enumerate(band_names, start=1):
        col = f"{prefix}_{bname}"
        df[col] = sample_raster(path, xs, ys, band=i)
    n_valid = df[f"{prefix}_{band_names[0]}"].notna().sum()
    print(f"  {prefix}: {n_valid}/{len(df)} valid ({100*n_valid/len(df):.1f}%)")

# ─────────────────────────────────────────────
# EXTRACT CATEGORICAL RASTERS (one-hot encode)
# ─────────────────────────────────────────────
print("\nExtracting categorical rasters...")
for fname in CATEGORICAL:
    path = os.path.join(COV_DIR, fname)
    prefix = fname.replace('.tif', '')
    if not os.path.exists(path):
        print(f"  WARNING: {fname} not found, skipping")
        continue
    raw = sample_raster(path, xs, ys)
    # One-hot encode — skip NaN and 0 (nodata/background)
    cats = [int(v) for v in np.unique(raw[~np.isnan(raw)]) if v != 0]
    for c in cats:
        df[f"{prefix}_{c}"] = (raw == c).astype(int)
    print(f"  {prefix}: {len(cats)} categories, {(~np.isnan(raw)).sum()}/{len(df)} valid")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print(f"\nTotal columns in output: {len(df.columns)}")
covar_cols = [c for c in df.columns if c not in [
    'point_id','source','flag','lat','lon',
    'total_cm_in_range','coverage_weight',
    'Fibric_pct','Hemic_pct','Sapric_pct','Mineral_pct','Unknown_pct','pct_sum'
]]
nan_counts = df[covar_cols].isna().sum()
high_nan = nan_counts[nan_counts > len(df) * 0.05]
if len(high_nan) > 0:
    print(f"\nCovariates with >5% NaN:")
    for col, n in high_nan.items():
        print(f"  {col}: {n} NaN ({100*n/len(df):.1f}%)")
else:
    print("  All covariates <5% NaN")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} rows x {len(df.columns)} columns to:\n  {OUTPUT_CSV}")
