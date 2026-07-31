#!/usr/bin/env python3
"""
Extract values from ONLY the new organic soils raster
Add to existing extracted data
"""

import pandas as pd
import numpy as np
from osgeo import gdal

# Paths
existing_csv = "/scratch.global/ocon0444/peat_modeling/00_data/processed/peat_depths_with_covariates_fixed.csv"
new_raster = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m/MN_organic_soils_classified_FIXED.tif"
output_csv = "/scratch.global/ocon0444/peat_modeling/00_data/processed/peat_depths_with_covariates_v2_fixed.csv"

print("Loading existing data...")
df = pd.read_csv(existing_csv, low_memory=False)
print(f"Loaded {len(df)} points with {len(df.columns)} columns")

print("\nExtracting from new organic soils raster...")
ds = gdal.Open(new_raster)
gt = ds.GetGeoTransform()
band = ds.GetRasterBand(1)
nodata = band.GetNoDataValue()

values = []
for idx, row in df.iterrows():
    x = row['ALBERS_X']
    y = row['ALBERS_Y']
    
    px = int((x - gt[0]) / gt[1])
    py = int((y - gt[3]) / gt[5])
    
    if 0 <= px < ds.RasterXSize and 0 <= py < ds.RasterYSize:
        value = band.ReadAsArray(px, py, 1, 1)[0, 0]
        if nodata is not None and value == nodata:
            values.append(0)  # NaN = no organic soil = 0
        else:
            values.append(value)
    else:
        values.append(0)

df['MN_organic_soils_classified_FIXED'] = values

print(f"\nExtracted values:")
print(f"  Unique values: {sorted(df['MN_organic_soils_classified_FIXED'].unique())}")
print(f"  Value counts:")
print(df['MN_organic_soils_classified_FIXED'].value_counts().sort_index())

# Save
df.to_csv(output_csv, index=False)
print(f"\nSaved to: {output_csv}")
print(f"Shape: {df.shape}")

ds = None
