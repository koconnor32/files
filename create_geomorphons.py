import sys
import os

wbt_dir = '/users/7/ocon0444/software/whitebox/WhiteboxTools_linux_amd64/WBT'
sys.path.insert(0, wbt_dir)
os.chdir(wbt_dir)

import whitebox_tools

wbt = whitebox_tools.WhiteboxTools()
wbt.set_verbose_mode(True)
wbt.set_compress_rasters(True)

dem = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m/minnesota_dem_10m.tif"
output = "/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m/geomorphons.tif"

print("Creating geomorphons...")
print(f"Input: {dem}")
print(f"Output: {output}")

wbt.geomorphons(dem, output, search=50, threshold=0.0, fdist=0, skip=0, forms=True)

print("Complete!")
