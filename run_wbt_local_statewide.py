#!/usr/bin/env python3
"""
Run WhiteboxTools local operations on full Minnesota DEM
"""

import sys
import os

# Setup WBT - MUST be in this exact order
wbt_dir = '/users/7/ocon0444/software/whitebox/WhiteboxTools_linux_amd64/WBT'
sys.path.insert(0, wbt_dir)
os.chdir(wbt_dir)  # CRITICAL: must chdir to WBT directory for license

import whitebox_tools

# Create WBT instance
wbt = whitebox_tools.WhiteboxTools()
wbt.set_verbose_mode(True)
wbt.set_compress_rasters(True)

# Paths
dem = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m/minnesota_dem_10m.tif"
output_dir = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m"

print("="*80)
print("WHITEBOXTOOLS - LOCAL OPERATIONS (STATEWIDE)")
print("="*80)
print(f"Input DEM: {dem}")
print(f"Output dir: {output_dir}")
print()

# Local operations
operations = [
    ("slope", "Slope"),
    ("aspect", "Aspect"),
    ("hillshade", "Hillshade"),
    ("planCurvature", "Plan Curvature"),
    ("profileCurvature", "Profile Curvature"),
    ("meanCurvature", "Mean Curvature"),
    ("maximalCurvature", "Maximal Curvature"),
]

for fname, display in operations:
    output = f"{output_dir}/{fname}.tif"
    
    print(f"\n{'='*60}")
    print(f"{display}")
    print('='*60)
    
    if os.path.exists(output):
        print("  ⚠ Already exists, skipping")
        continue
    
    try:
        if fname == "slope":
            wbt.slope(dem, output, units="degrees")
        elif fname == "aspect":
            wbt.aspect(dem, output)
        elif fname == "hillshade":
            wbt.hillshade(dem, output, azimuth=315.0, altitude=45.0)
        elif fname == "planCurvature":
            wbt.plan_curvature(dem, output)
        elif fname == "profileCurvature":
            wbt.profile_curvature(dem, output)
        elif fname == "meanCurvature":
            wbt.mean_curvature(dem, output)
        elif fname == "maximalCurvature":
            wbt.maximal_curvature(dem, output)
        
        print(f"  ✓ Complete")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
