#!/usr/bin/env python3
"""
Run WhiteboxTools flow operations on a single HUC8 watershed
This script is called by SBATCH array job
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

# Get HUC8 code from command line argument
if len(sys.argv) < 2:
    print("Usage: python run_wbt_flow_by_huc8.py <HUC8_CODE>")
    sys.exit(1)

huc8_code = sys.argv[1]

# Paths
dem_dir = "/scratch.global/ocon0444/peat_modeling/00_data/dem_clipped_by_huc8"
output_dir = f"/scratch.global/ocon0444/peat_modeling/00_data/wbt_outputs_by_huc8/{huc8_code}"
dem = f"{dem_dir}/dem_huc8_{huc8_code}.tif"

os.makedirs(output_dir, exist_ok=True)

print("="*80)
print(f"WHITEBOXTOOLS - FLOW OPERATIONS FOR HUC8 {huc8_code}")
print("="*80)
print(f"Input DEM: {dem}")
print(f"Output dir: {output_dir}")
print()

if not os.path.exists(dem):
    print(f"ERROR: DEM not found: {dem}")
    sys.exit(1)

# Flow-based operations
try:
    # 1. Breach depressions
    print("\n[1/11] Breach Depressions")
    breached = f"{output_dir}/breached_dem.tif"
    if not os.path.exists(breached):
        wbt.breach_depressions(dem, breached, max_depth=None, max_length=None, flat_increment=None)
        print("  ✓ Complete")
    else:
        print("  ⚠ Already exists")
    
    # 2. D8 Flow Accumulation
    print("\n[2/11] D8 Flow Accumulation")
    d8_flow = f"{output_dir}/d8FlowAccumulation.tif"
    if not os.path.exists(d8_flow):
        wbt.d8_flow_accumulation(breached, d8_flow, out_type="cells")
        print("  ✓ Complete")
    else:
        print("  ⚠ Already exists")
    
    # 3. DInf Flow Accumulation
    print("\n[3/11] DInf Flow Accumulation")
    dinf_flow = f"{output_dir}/dInfFlowAccumulation.tif"
    if not os.path.exists(dinf_flow):
        wbt.d_inf_flow_accumulation(breached, dinf_flow, out_type="Specific Contributing Area")
        print("  ✓ Complete")
    else:
        print("  ⚠ Already exists")
    
    # 4. Wetness Index
    print("\n[4/11] Wetness Index")
    twi = f"{output_dir}/wetnessIndex.tif"
    if not os.path.exists(twi):
        wbt.wetness_index(dinf_flow, breached, twi)
        print("  ✓ Complete")
    else:
        print("  ⚠ Already exists")
    
    # 5-7. Deviation from Mean Elevation (multiple scales)
    for radius in [4, 8, 16]:
        print(f"\n[{4+radius//4}/11] Deviation from Mean Elevation ({radius}m)")
        output = f"{output_dir}/devfrommeanelev_{radius}m.tif"
        if not os.path.exists(output):
            wbt.dev_from_mean_elev(breached, output, filterx=radius, filtery=radius)
            print("  ✓ Complete")
        else:
            print("  ⚠ Already exists")
    
    # 8. Difference from Mean Elevation
    print("\n[8/11] Difference from Mean Elevation")
    diff_mean = f"{output_dir}/diffFromMeanElev.tif"
    if not os.path.exists(diff_mean):
        wbt.diff_from_mean_elev(breached, diff_mean, filterx=11, filtery=11)
        print("  ✓ Complete")
    else:
        print("  ⚠ Already exists")
    
    # 9-11. Relative Topographic Position (multiple scales)
    for radius in [4, 8, 16]:
        print(f"\n[{8+radius//4}/11] Relative Topographic Position ({radius}m)")
        output = f"{output_dir}/relativeTopographicPosition_{radius}m.tif"
        if not os.path.exists(output):
            wbt.relative_topographic_position(breached, output, filterx=radius, filtery=radius)
            print("  ✓ Complete")
        else:
            print("  ⚠ Already exists")
    
    print("\n" + "="*80)
    print(f"HUC8 {huc8_code} COMPLETE")
    print("="*80)
    
except Exception as e:
    print(f"\nERROR processing HUC8 {huc8_code}: {e}")
    sys.exit(1)
