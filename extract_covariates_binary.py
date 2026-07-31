#!/usr/bin/env python3
"""
Covariate extraction for binary peat classification modeling.
Extracts values from 29 rasters at ~60K point locations.
Handles: single-band, multi-band, and categorical rasters.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# ARGUMENTS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Extract covariate values for peat point dataset.")
parser.add_argument("--input",  required=True,  help="Path to input points CSV")
parser.add_argument("--output", required=True,  help="Path to output CSV")
args = parser.parse_args()

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
POINTS_CSV   = args.input
COV_DIR      = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m/"
OUTPUT_CSV   = args.output

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
    # "wetnessIndex.tif",  # EXCLUDED - mosaic has 45% NoData gaps, add back after fix
    "devfrommeanelev_4m.tif",
    "devfrommeanelev_8m.tif",
    "devfrommeanelev_16m.tif",
    "diffFromMeanElev.tif",
    "relativeTopographicPosition_4m.tif",
    "relativeTopographicPosition_8m.tif",
    "relativeTopographicPosition_16m.tif",
    "dist_to_water_10m.tif",
    "dist_to_stream_10m.tif",
    "prism_ppt_mn.tif",
    "prism_tmax_july_mn.tif",
    "prism_tmean_mn.tif",
    "prism_tmin_january_mn.tif",
]

# Categorical rasters → will be one-hot encoded
CATEGORICAL = [
    "10m_quaternary_geology.tif",
    "pennockLandformClass.tif",
]

# Multi-band rasters → (filename, band_prefix, band_names)
SENTINEL2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12", "NDVI", "SWDI"]
MULTIBAND = [
    ("s2_spring_12bands.tif",  "s2_spring",  SENTINEL2_BANDS),
    ("s2_summer_12bands.tif",  "s2_summer",  SENTINEL2_BANDS),
    ("s2_fall_12bands.tif",    "s2_fall",    SENTINEL2_BANDS),
]

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def load_points(csv_path):
    print(f"Loading points from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} points")
    print(f"  Columns: {list(df.columns)}")
    return df


def reproject_points(df, src_epsg=4326, dst_epsg=5070):
    """Convert lat/long (WGS84) to EPSG:5070 (Conus Albers)."""
    print(f"  Reprojecting from EPSG:{src_epsg} to EPSG:{dst_epsg}...")
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
    x, y = transformer.transform(df["long"].values, df["lat"].values)
    df = df.copy()
    df["x_5070"] = x
    df["y_5070"] = y
    return df


def extract_single_band(df, raster_path, col_name):
    """Extract values from a single-band raster."""
    coords = list(zip(df["x_5070"], df["y_5070"]))
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        values = np.array([v[0] for v in src.sample(coords)])
        if nodata is not None:
            values = values.astype(float)
            values[values == nodata] = np.nan
    df[col_name] = values
    return df


def extract_multiband(df, raster_path, prefix, band_names):
    """Extract all bands from a multi-band raster."""
    coords = list(zip(df["x_5070"], df["y_5070"]))
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        n_bands = src.count
        assert n_bands == len(band_names), \
            f"{raster_path}: expected {len(band_names)} bands, got {n_bands}"
        # Sample returns array of shape (n_points, n_bands)
        sampled = np.array(list(src.sample(coords)))  # (n_points, n_bands)
        sampled = sampled.astype(float)
        if nodata is not None:
            sampled[sampled == nodata] = np.nan
    for i, bname in enumerate(band_names):
        df[f"{prefix}_{bname}"] = sampled[:, i]
    return df


def extract_categorical_onehot(df, raster_path, prefix):
    """Extract categorical raster and one-hot encode."""
    coords = list(zip(df["x_5070"], df["y_5070"]))
    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        values = np.array([v[0] for v in src.sample(coords)])
        if nodata is not None:
            values = values.astype(float)
            values[values == nodata] = np.nan

    raw_col = f"{prefix}_raw"
    df[raw_col] = values

    # One-hot encode (drop_first=False to keep all classes; prefix handles naming)
    dummies = pd.get_dummies(df[raw_col].astype("Int64"), prefix=prefix)
    df = pd.concat([df, dummies], axis=1)
    df.drop(columns=[raw_col], inplace=True)
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 1. Load points
    df = load_points(POINTS_CSV)

    # 2. Reproject to EPSG:5070
    df = reproject_points(df)

    # 3. Extract single-band rasters
    print("\n--- Single-band rasters ---")
    for fname in SINGLE_BAND:
        fpath = os.path.join(COV_DIR, fname)
        col = fname.replace(".tif", "")
        if not os.path.exists(fpath):
            print(f"  [SKIP - not found] {fname}")
            continue
        print(f"  Extracting: {fname} → {col}")
        df = extract_single_band(df, fpath, col)

    # 4. Extract multi-band Sentinel-2 rasters
    print("\n--- Multi-band rasters (Sentinel-2) ---")
    for fname, prefix, bands in MULTIBAND:
        fpath = os.path.join(COV_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP - not found] {fname}")
            continue
        print(f"  Extracting: {fname} → {prefix}_B02 ... {prefix}_SWDI")
        df = extract_multiband(df, fpath, prefix, bands)

    # 5. Extract categorical rasters with one-hot encoding
    print("\n--- Categorical rasters (one-hot) ---")
    for fname in CATEGORICAL:
        fpath = os.path.join(COV_DIR, fname)
        prefix = fname.replace(".tif", "").replace("10m_", "")
        if not os.path.exists(fpath):
            print(f"  [SKIP - not found] {fname}")
            continue
        print(f"  Extracting & encoding: {fname} → {prefix}_*")
        df = extract_categorical_onehot(df, fpath, prefix)

    # 6. Drop rows with any NaN in covariate columns
    # Keep original columns + extracted; drop rows where covariates are null
    orig_cols = ["lat", "long", "depb", "peat_binary"]
    covariate_cols = [c for c in df.columns if c not in orig_cols + ["x_5070", "y_5070"]]
    
    print(f"\n--- Cleaning ---")
    print(f"  Rows before NaN removal: {len(df):,}")
    df_clean = df.dropna(subset=covariate_cols)
    print(f"  Rows after NaN removal:  {len(df_clean):,}")
    print(f"  Dropped: {len(df) - len(df_clean):,} rows")

    # 7. Final column order: id info, target, covariates
    final_cols = ["lat", "long", "depb", "peat_binary"] + covariate_cols
    df_out = df_clean[final_cols].reset_index(drop=True)
    df_out.insert(0, "point_id", df_out.index)

    # 8. Save
    print(f"\n--- Saving ---")
    print(f"  Output shape: {df_out.shape}")
    print(f"  Output columns ({len(df_out.columns)}): {list(df_out.columns[:10])} ...")
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved to: {OUTPUT_CSV}")
    print("\nDone!")


if __name__ == "__main__":
    main()
