#!/usr/bin/env python3
"""
Complete distance raster creation workflow:
1. Reproject vectors from UTM to Albers
2. Create distance rasters
3. Mask to Minnesota state boundary
"""

from osgeo import gdal, ogr, osr
import os

# Paths
vector_dir = "/scratch.global/ocon0444/peat_modeling/00_data/vector_data"
output_dir = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m"
ref_raster = f"{output_dir}/minnesota_dem_10m.tif"
mn_boundary = "/scratch.global/ocon0444/peat_modeling/00_data/boundary/mn_state_boundary_albers.shp"

# Get reference raster properties
ref_ds = gdal.Open(ref_raster)
ref_gt = ref_ds.GetGeoTransform()
ref_proj = ref_ds.GetProjection()
cols = ref_ds.RasterXSize
rows = ref_ds.RasterYSize
ref_ds = None

# Define CRS
target_srs = osr.SpatialReference()
target_srs.ImportFromWkt(ref_proj)

source_srs = osr.SpatialReference()
source_srs.ImportFromEPSG(26915)  # UTM Zone 15N

print("="*80)
print("CREATING DISTANCE RASTERS - COMPLETE WORKFLOW")
print("="*80)
print(f"Reference: {cols} x {rows} @ 10m resolution")
print(f"Source CRS: UTM Zone 15N → Target: NAD83 Conus Albers")
print(f"Masking to: Minnesota state boundary")

# Define rasters to create
distance_layers = [
    ("HydroFeatures.shp", "dist_to_water_10m.tif", "Distance to lakes/ponds"),
    ("RiversStreams.shp", "dist_to_stream_10m.tif", "Distance to rivers/streams"),
    ("MnDOT_Roadway_Routes_in_Minnesota.shp", "dist_to_road_detailed_10m.tif", "Distance to roads"),
]

for vector_name, output_name, description in distance_layers:
    vector_path = f"{vector_dir}/{vector_name}"
    temp_output = f"{output_dir}/{output_name}.temp.tif"
    final_output = f"{output_dir}/{output_name}"
    
    print(f"\n{'='*80}")
    print(f"{description}")
    print('='*80)
    print(f"  Input: {vector_name}")
    print(f"  Output: {output_name}")
    
    if not os.path.exists(vector_path):
        print(f"  ❌ Vector file not found, skipping")
        continue
    
    # Step 1: Reproject vector
    print(f"\n  [1/4] Reprojecting from UTM to Albers...")
    vector_ds = ogr.Open(vector_path)
    src_layer = vector_ds.GetLayer()
    feature_count = src_layer.GetFeatureCount()
    print(f"        Features: {feature_count:,}")
    
    mem_driver = ogr.GetDriverByName('Memory')
    mem_ds = mem_driver.CreateDataSource('memData')
    mem_layer = mem_ds.CreateLayer('reprojected', target_srs, src_layer.GetGeomType())
    
    transform = osr.CoordinateTransformation(source_srs, target_srs)
    
    src_layer.ResetReading()
    skipped = 0
    for feature in src_layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            skipped += 1
            continue
        
        geom = geom.Clone()
        geom.Transform(transform)
        
        out_feature = ogr.Feature(mem_layer.GetLayerDefn())
        out_feature.SetGeometry(geom)
        mem_layer.CreateFeature(out_feature)
        out_feature = None
    
    vector_ds = None
    print(f"        Reprojected: {mem_layer.GetFeatureCount():,} features")
    if skipped > 0:
        print(f"        Skipped: {skipped} null geometries")
    
    # Step 2: Rasterize
    print(f"\n  [2/4] Rasterizing features...")
    temp_raster = "/vsimem/temp_features.tif"
    
    driver = gdal.GetDriverByName('GTiff')
    temp_ds = driver.Create(temp_raster, cols, rows, 1, gdal.GDT_Byte)
    temp_ds.SetGeoTransform(ref_gt)
    temp_ds.SetProjection(ref_proj)
    temp_band = temp_ds.GetRasterBand(1)
    temp_band.Fill(0)
    
    gdal.RasterizeLayer(temp_ds, [1], mem_layer, burn_values=[1])
    temp_ds.FlushCache()
    mem_ds = None
    
    # Step 3: Calculate distance
    print(f"\n  [3/4] Calculating distance transform...")
    print(f"        (This takes 5-10 minutes per layer)")
    
    dist_ds = driver.Create(temp_output, cols, rows, 1, gdal.GDT_Float32,
                           options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
    dist_ds.SetGeoTransform(ref_gt)
    dist_ds.SetProjection(ref_proj)
    dist_band = dist_ds.GetRasterBand(1)
    
    temp_ds = gdal.Open(temp_raster)
    temp_band = temp_ds.GetRasterBand(1)
    
    gdal.ComputeProximity(temp_band, dist_band, ["DISTUNITS=GEO", "VALUES=1"])
    
    dist_ds.FlushCache()
    dist_ds = None
    temp_ds = None
    gdal.Unlink(temp_raster)
    
    # Step 4: Mask to Minnesota boundary
    print(f"\n  [4/4] Masking to Minnesota boundary...")
    
    gdal.Warp(
        final_output,
        temp_output,
        cutlineDSName=mn_boundary,
        cropToCutline=True,
        dstNodata=-9999,
        creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
    )
    
    # Remove temp file
    os.remove(temp_output)
    
    print(f"\n  ✓ Complete: {output_name}")

print("\n" + "="*80)
print("ALL DISTANCE RASTERS CREATED")
print("="*80)
print(f"\nOutput directory: {output_dir}")
print(f"Files created:")
print(f"  - dist_to_water_10m.tif")
print(f"  - dist_to_stream_10m.tif")
print(f"  - dist_to_road_detailed_10m.tif")
print(f"\nAll masked to Minnesota state boundary")
print("="*80)
