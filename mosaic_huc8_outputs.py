#!/usr/bin/env python3
"""
Mosaic 127 HUC8 WhiteboxTools outputs into statewide rasters
Handles overlapping buffers and aligns to gNATSGO grid
"""

import os
import subprocess
import glob
from osgeo import gdal

# Paths
huc8_dir = "/scratch.global/ocon0444/peat_modeling/00_data/wbt_outputs_by_huc8"
output_dir = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m"
ref_raster = f"{output_dir}/gNATSGO_MN_26915.tif"

# Get reference grid info
ref_ds = gdal.Open(ref_raster)
ref_gt = ref_ds.GetGeoTransform()
ref_size = (ref_ds.RasterXSize, ref_ds.RasterYSize)
ref_ds = None

print("="*80)
print("MOSAICKING HUC8 OUTPUTS TO STATEWIDE RASTERS")
print("="*80)
print(f"Input: {huc8_dir}")
print(f"Output: {output_dir}")
print(f"Reference grid: {ref_size[0]} x {ref_size[1]}")
print(f"Extent: {ref_gt[0]}, {ref_gt[3]} to {ref_gt[0] + ref_size[0]*ref_gt[1]}, {ref_gt[3] + ref_size[1]*ref_gt[5]}")
print()

# List of outputs to mosaic
outputs = [
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
]

for idx, output_name in enumerate(outputs, 1):
    print(f"\n{'='*80}")
    print(f"[{idx}/11] {output_name}")
    print('='*80)
    
    final_output = f"{output_dir}/{output_name}"
    
    if os.path.exists(final_output):
        print(f"  ⚠ Already exists, skipping")
        continue
    
    # Find all HUC8 tiles for this output
    tile_pattern = f"{huc8_dir}/*/{output_name}"
    tiles = glob.glob(tile_pattern)
    
    if len(tiles) == 0:
        print(f"  ✗ No tiles found matching: {tile_pattern}")
        continue
    
    print(f"  Found {len(tiles)} tiles")
    
    # Create VRT (virtual mosaic) first - fast and handles overlaps
    vrt_file = f"/tmp/temp_mosaic_{output_name}.vrt"
    temp_output = f"/tmp/temp_{output_name}"
    
    print(f"  [1/3] Building VRT mosaic...")
    
    try:
        # Build VRT
        vrt_ds = gdal.BuildVRT(
            vrt_file,
            tiles,
            resolution='highest',
            resampleAlg='nearest',
            addAlpha=False
        )
        vrt_ds = None
        
        print(f"  [2/3] Warping to gNATSGO grid...")
        
        # Warp VRT to match gNATSGO grid exactly
        warp_options = gdal.WarpOptions(
            format='GTiff',
            outputBounds=(ref_gt[0], 
                         ref_gt[3] + ref_size[1]*ref_gt[5],
                         ref_gt[0] + ref_size[0]*ref_gt[1],
                         ref_gt[3]),
            xRes=10,
            yRes=10,
            targetAlignedPixels=True,
            resampleAlg='bilinear',
            creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
        )
        
        gdal.Warp(temp_output, vrt_file, options=warp_options)
        
        print(f"  [3/3] Final alignment check...")
        
        # Ensure EXACT alignment with -ts flag
        subprocess.run([
            'gdalwarp',
            '-ts', str(ref_size[0]), str(ref_size[1]),
            '-te', str(ref_gt[0]), 
                   str(ref_gt[3] + ref_size[1]*ref_gt[5]),
                   str(ref_gt[0] + ref_size[0]*ref_gt[1]),
                   str(ref_gt[3]),
            '-r', 'bilinear',
            '-co', 'COMPRESS=LZW',
            '-co', 'TILED=YES',
            '-co', 'BIGTIFF=YES',
            temp_output,
            final_output
        ], check=True)
        
        # Cleanup
        if os.path.exists(vrt_file):
            os.remove(vrt_file)
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        # Verify output
        verify_ds = gdal.Open(final_output)
        verify_size = (verify_ds.RasterXSize, verify_ds.RasterYSize)
        verify_gt = verify_ds.GetGeoTransform()
        verify_ds = None
        
        if verify_size == ref_size and verify_gt == ref_gt:
            print(f"  ✓ Complete - Aligned to gNATSGO grid")
        else:
            print(f"  ⚠ Warning: Size or alignment mismatch")
            print(f"     Expected: {ref_size}, Got: {verify_size}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        # Cleanup on error
        for f in [vrt_file, temp_output]:
            if os.path.exists(f):
                os.remove(f)

print("\n" + "="*80)
print("MOSAIC COMPLETE")
print("="*80)
print(f"Created 11 statewide rasters in: {output_dir}")
print("All aligned to gNATSGO grid")
print("="*80)
