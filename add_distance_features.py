#!/usr/bin/env python3
"""
Add distance raster values to existing dataset
NO one-hot encoding - these are continuous values!
"""

import pandas as pd
import numpy as np
from osgeo import gdal

# Paths
input_csv = "/scratch.global/ocon0444/peat_modeling/00_data/processed/peat_depths_processed_v3_with_NPC.csv"
output_csv = "/scratch.global/ocon0444/peat_modeling/00_data/processed/peat_depths_processed_v4_with_distances.csv"
raster_dir = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m"

distance_rasters = {
    'dist_to_water_m': 'dist_to_water_10m.tif',
    'dist_to_stream_m': 'dist_to_stream_10m.tif',
    'dist_to_road_m': 'dist_to_road_detailed_10m.tif'
}

print("="*80)
print("ADDING DISTANCE FEATURES")
print("="*80)

# Load existing data
print("\nLoading existing data...")
df = pd.read_csv(input_csv, low_memory=False)
print(f"Loaded {len(df)} points with {len(df.columns)} columns")

# Extract from each raster
for col_name, raster_name in distance_rasters.items():
    raster_path = f"{raster_dir}/{raster_name}"
    
    print(f"\nExtracting {col_name} from {raster_name}...")
    
    # Open raster
    ds = gdal.Open(raster_path)
    if ds is None:
        print(f"  ❌ Could not open {raster_path}")
        continue
        
    band = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    
    # Extract values
    values = []
    for idx, row in df.iterrows():
        x = row['ALBERS_X']
        y = row['ALBERS_Y']
        
        # Convert coordinates to pixel indices
        px = int((x - gt[0]) / gt[1])
        py = int((y - gt[3]) / gt[5])
        
        # Read value
        try:
            value = band.ReadAsArray(px, py, 1, 1)[0, 0]
            # Check for NoData
            nodata = band.GetNoDataValue()
            if nodata is not None and value == nodata:
                value = np.nan
            values.append(value)
        except:
            values.append(np.nan)
    
    ds = None
    
    # Add to dataframe
    df[col_name] = values
    
    # Summary
    print(f"  Min: {df[col_name].min():.2f} m")
    print(f"  Max: {df[col_name].max():.2f} m")
    print(f"  Mean: {df[col_name].mean():.2f} m")
    print(f"  Missing: {df[col_name].isna().sum()}")

# Drop rows with missing distance values
print(f"\n{'='*80}")
print("HANDLING MISSING VALUES")
print("="*80)

rows_before = len(df)
df = df.dropna(subset=list(distance_rasters.keys()))
rows_after = len(df)

print(f"Rows before: {rows_before}")
print(f"Rows after: {rows_after}")
print(f"Dropped: {rows_before - rows_after} ({(rows_before-rows_after)/rows_before*100:.1f}%)")

# Save
print(f"\nSaving to {output_csv}...")
df.to_csv(output_csv, index=False)
print(f"Final shape: {df.shape}")
print(f"Total features: {df.shape[1] - 7} (excluding metadata)")
print("="*80)
