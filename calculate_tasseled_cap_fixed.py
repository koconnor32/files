#!/usr/bin/env python3
"""
Calculate Tasseled Cap using gdal_calc.py (more reliable than rasterio windowing)
"""

import subprocess
import os

# Tasseled Cap coefficients for Sentinel-2
TC_COEFF = {
    'brightness': {'B2': 0.3029, 'B3': 0.2786, 'B4': 0.4733, 'B8': 0.5599, 'B11': 0.508, 'B12': 0.1872},
    'greenness': {'B2': -0.2941, 'B3': -0.243, 'B4': -0.5424, 'B8': 0.7276, 'B11': 0.0713, 'B12': -0.1608},
    'wetness': {'B2': 0.1511, 'B3': 0.1973, 'B4': 0.3283, 'B8': 0.3407, 'B11': -0.7117, 'B12': -0.4559}
}

def calculate_tc_band(input_file, output_file, tc_type):
    """Calculate one Tasseled Cap component using gdal_calc.py"""
    coeff = TC_COEFF[tc_type]
    
    # Build gdal_calc command
    calc_expr = (
        f"{coeff['B2']}*A + {coeff['B3']}*B + {coeff['B4']}*C + "
        f"{coeff['B8']}*D + {coeff['B11']}*E + {coeff['B12']}*F"
    )
    
    cmd = [
        'gdal_calc.py',
        '-A', input_file, '--A_band=1',   # B2 (Blue)
        '-B', input_file, '--B_band=2',   # B3 (Green)
        '-C', input_file, '--C_band=3',   # B4 (Red)
        '-D', input_file, '--D_band=7',   # B8 (NIR)
        '-E', input_file, '--E_band=9',   # B11 (SWIR1)
        '-F', input_file, '--F_band=10',  # B12 (SWIR2)
        '--outfile=' + output_file,
        '--calc=' + calc_expr,
        '--type=Float32',
        '--co=COMPRESS=LZW',
        '--co=TILED=YES',
        '--co=BIGTIFF=YES',
        '--quiet'
    ]
    
    print(f"  Calculating {tc_type}...")
    subprocess.run(cmd, check=True)

def stack_bands(input_12band, brightness, greenness, wetness, output_15band):
    """Stack original 12 bands + 3 TC bands into one file"""
    print("  Stacking all 15 bands...")
    
    cmd = [
        'gdal_merge.py',
        '-separate',
        '-o', output_15band,
        '-co', 'COMPRESS=LZW',
        '-co', 'TILED=YES',
        '-co', 'BIGTIFF=YES',
        input_12band,  # This adds bands 1-12
        brightness,    # Band 13
        greenness,     # Band 14
        wetness        # Band 15
    ]
    
    subprocess.run(cmd, check=True)

def process_season(season_name, input_file):
    """Process one season"""
    print(f"\n{'='*60}")
    print(f"Processing {season_name}")
    print(f"{'='*60}")
    
    # Temporary TC band files
    brightness_file = f"temp_{season_name}_brightness.tif"
    greenness_file = f"temp_{season_name}_greenness.tif"
    wetness_file = f"temp_{season_name}_wetness.tif"
    
    # Final output
    output_file = input_file.replace('12bands', '15bands')
    
    if os.path.exists(output_file):
        print(f"SKIPPED: {output_file} already exists")
        return
    
    try:
        # Calculate TC bands
        calculate_tc_band(input_file, brightness_file, 'brightness')
        calculate_tc_band(input_file, greenness_file, 'greenness')
        calculate_tc_band(input_file, wetness_file, 'wetness')
        
        # Stack everything together
        stack_bands(input_file, brightness_file, greenness_file, wetness_file, output_file)
        
        # Clean up temp files
        os.remove(brightness_file)
        os.remove(greenness_file)
        os.remove(wetness_file)
        
        print(f"  SUCCESS: Created {output_file}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        # Clean up temp files on error
        for f in [brightness_file, greenness_file, wetness_file]:
            if os.path.exists(f):
                os.remove(f)

def main():
    data_dir = '/scratch.global/ocon0444/peat_modeling/00_data/covariates_10m/'
    os.chdir(data_dir)
    
    seasons = [
        ('Spring', 's2_spring_12bands.tif'),
        ('Summer', 's2_summer_12bands.tif'),
        ('Fall', 's2_fall_12bands.tif')
    ]
    
    for season_name, input_file in seasons:
        process_season(season_name, input_file)
    
    print(f"\n{'='*60}")
    print("DONE! All seasons processed.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
