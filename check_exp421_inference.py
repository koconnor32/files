"""
check_exp421_inference.py
Quick diagnostic on exp421 output rasters.
Checks value ranges, % over 100, sum of Fibric+Hemic+Sapric per pixel, etc.
"""
import os
import numpy as np
import rasterio

BASE    = '/scratch.global/ocon0444/peat_modeling'
OUT_DIR = os.path.join(BASE, '04_predictions/exp421')

REGIONS  = ['redlake', 'SE', 'SW']
TARGETS  = ['fibric', 'hemic', 'sapric']
NODATA   = -9999.0

for region in REGIONS:
    print(f"\n{'='*60}")
    print(f"Region: {region}")
    print('='*60)

    arrays = {}
    for target in TARGETS:
        path = os.path.join(OUT_DIR, f'exp421_{target}_{region}.tif')
        with rasterio.open(path) as src:
            data = src.read(1).astype(float)
            data[data == NODATA] = np.nan
        arrays[target] = data
        valid = data[~np.isnan(data)]
        pct_over_100 = 100 * (valid > 100).sum() / len(valid) if len(valid) > 0 else 0
        pct_under_0  = 100 * (valid < 0).sum()   / len(valid) if len(valid) > 0 else 0
        print(f"  {target:<10} | n_valid={len(valid):>8,} | "
              f"min={valid.min():>7.1f}  mean={valid.mean():>6.1f}  "
              f"max={valid.max():>7.1f} | "
              f">100: {pct_over_100:.1f}%  <0: {pct_under_0:.1f}%")

    # Check organic_sum raster
    sum_path = os.path.join(OUT_DIR, f'exp421_organic_sum_{region}.tif')
    with rasterio.open(sum_path) as src:
        sum_data = src.read(1).astype(float)
        sum_data[sum_data == NODATA] = np.nan
    valid_sum = sum_data[~np.isnan(sum_data)]
    print(f"  {'organic_sum':<10} | n_valid={len(valid_sum):>8,} | "
          f"min={valid_sum.min():>7.1f}  mean={valid_sum.mean():>6.1f}  "
          f"max={valid_sum.max():>7.1f}")

    # Recompute sum from individual rasters and compare
    recomputed = np.nansum([arrays[t] for t in TARGETS], axis=0)
    # Only where all 3 are valid
    all_valid  = ~(np.isnan(arrays['fibric']) | np.isnan(arrays['hemic']) | np.isnan(arrays['sapric']))
    rc_vals    = recomputed[all_valid]

    print(f"\n  Recomputed F+H+S sum (where all valid):")
    print(f"    n pixels  : {all_valid.sum():,}")
    print(f"    mean sum  : {rc_vals.mean():.1f}%")
    print(f"    min sum   : {rc_vals.min():.1f}%")
    print(f"    max sum   : {rc_vals.max():.1f}%")
    print(f"    >100%     : {(rc_vals > 100).sum():,} pixels ({100*(rc_vals>100).mean():.1f}%)")
    print(f"    >150%     : {(rc_vals > 150).sum():,} pixels ({100*(rc_vals>150).mean():.1f}%)")
    print(f"    <10%      : {(rc_vals < 10).sum():,}  pixels ({100*(rc_vals<10).mean():.1f}%)")

    # Percentile distribution of the sum
    pcts = np.percentile(rc_vals, [1, 5, 25, 50, 75, 95, 99])
    labels = ['p1','p5','p25','p50','p75','p95','p99']
    print(f"    Percentiles: " + "  ".join(f"{l}={v:.1f}" for l, v in zip(labels, pcts)))

    # Per-target mean where peat mask is active
    print(f"\n  Mean composition where peat mask active:")
    total_mean = 0
    for target in TARGETS:
        v = arrays[target][all_valid]
        print(f"    {target}: {v.mean():.1f}%")
        total_mean += v.mean()
    print(f"    sum of means: {total_mean:.1f}%  (remaining ~{100-total_mean:.1f}% is Mineral/Unknown)")
