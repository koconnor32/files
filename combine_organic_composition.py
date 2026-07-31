"""
combine_organic_composition.py

Combines peat inventory and NASIS organic-only horizon composition datasets
into a single clean CSV for hydraulic conductivity modeling.

Outputs to:
    /scratch.global/ocon0444/peat_modeling/00_data/point_data/organic_composition_0_50cm_combined.csv

Steps:
    1. Load peat inventory (_all) and NASIS (ORGANIC_ONLY)
    2. Reconcile columns (collapse NASIS unknown columns, standardize lat/lon/id names)
    3. Add source column and coverage_weight (total_cm_in_range / 50, capped at 1.0)
    4. Add pct_sum QA column
    5. Drop rows with missing coordinates
    6. Clip to MN state boundary
    7. Print summary for review
    8. Save combined CSV
"""

import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
PEAT_INV_PATH  = '/scratch.global/ocon0444/peat_inventory/horizon_composition_0_50cm_all.csv'
NASIS_PATH     = '/scratch.global/ocon0444/NASIS_points/horizon_composition_0_50cm_ORGANIC_ONLY.csv'
MN_BOUNDARY    = '/scratch.global/ocon0444/peat_modeling/00_data/boundary/mn_state_boundary_albers.shp'
OUT_PATH       = '/scratch.global/ocon0444/peat_modeling/00_data/point_data/organic_composition_0_50cm_combined.csv'

# ── Target columns in final output ─────────────────────────────────────────────
KEEP_COLS = [
    'point_id', 'source', 'flag',
    'lat', 'lon',
    'total_cm_in_range', 'coverage_weight',
    'Fibric_pct', 'Hemic_pct', 'Sapric_pct', 'Mineral_pct', 'Unknown_pct',
    'pct_sum'
]

# ── Load peat inventory ────────────────────────────────────────────────────────
print("Loading peat inventory...")
pi = pd.read_csv(PEAT_INV_PATH)
print(f"  Raw rows: {len(pi)}")

pi = pi.rename(columns={'profile_id': 'point_id'})
pi['source'] = 'peat_inventory'
# Unknown_pct already named correctly in peat inventory

pi = pi[['point_id', 'source', 'flag', 'lat', 'lon',
         'total_cm_in_range', 'Fibric_pct', 'Hemic_pct',
         'Sapric_pct', 'Mineral_pct', 'Unknown_pct']]

# ── Load NASIS organic only ────────────────────────────────────────────────────
print("Loading NASIS organic only...")
nasis = pd.read_csv(NASIS_PATH)
print(f"  Raw rows: {len(nasis)}")

nasis = nasis.rename(columns={
    'upedonid':              'point_id',
    'latstddecimaldegrees':  'lat',
    'longstddecimaldegrees': 'lon'
})
nasis['source'] = 'NASIS'

# Collapse two unknown columns into one to match peat inventory schema
nasis['Unknown_pct'] = nasis['Organic_Unknown_pct'].fillna(0) + nasis['Unknown_horizon_pct'].fillna(0)

nasis = nasis[['point_id', 'source', 'flag', 'lat', 'lon',
               'total_cm_in_range', 'Fibric_pct', 'Hemic_pct',
               'Sapric_pct', 'Mineral_pct', 'Unknown_pct']]

# ── Combine ────────────────────────────────────────────────────────────────────
print("\nCombining datasets...")
combined = pd.concat([pi, nasis], ignore_index=True)
print(f"  Combined rows: {len(combined)}")

# ── Coverage weight ────────────────────────────────────────────────────────────
combined['coverage_weight'] = (combined['total_cm_in_range'] / 50.0).clip(upper=1.0)

# ── QA sum column ──────────────────────────────────────────────────────────────
combined['pct_sum'] = combined[['Fibric_pct', 'Hemic_pct', 'Sapric_pct',
                                 'Mineral_pct', 'Unknown_pct']].sum(axis=1)

# ── Drop missing coordinates ───────────────────────────────────────────────────
n_before = len(combined)
combined = combined.dropna(subset=['lat', 'lon'])
n_dropped = n_before - len(combined)
print(f"  Dropped {n_dropped} rows with missing lat/lon")

# ── Clip to MN bounding box ────────────────────────────────────────────────────
# MN approximate bounds: lat 43.5-49.4, lon -97.25 to -89.45
print("\nClipping to MN bounding box...")
MN_LAT_MIN, MN_LAT_MAX =  43.5,  49.4
MN_LON_MIN, MN_LON_MAX = -97.25, -89.45

in_mn = combined[
    (combined['lat'] >= MN_LAT_MIN) & (combined['lat'] <= MN_LAT_MAX) &
    (combined['lon'] >= MN_LON_MIN) & (combined['lon'] <= MN_LON_MAX)
].copy()

print(f"  Rows before MN clip: {len(combined)}")
print(f"  Rows after MN clip:  {len(in_mn)}")
print(f"  Removed (out of state): {len(combined) - len(in_mn)}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n── Source breakdown ──────────────────────────────────────────────────────")
print(in_mn['source'].value_counts().to_string())

print("\n── Flag breakdown (top 20) ───────────────────────────────────────────────")
print(in_mn['flag'].value_counts().head(20).to_string())

print("\n── Coverage weight distribution ──────────────────────────────────────────")
print(in_mn['coverage_weight'].describe().to_string())

print("\n── Pct sum stats (should be ~100 for all rows) ───────────────────────────")
print(in_mn['pct_sum'].describe().to_string())
bad_sum = in_mn[(in_mn['pct_sum'] < 95) | (in_mn['pct_sum'] > 105)]
print(f"  Rows with pct_sum outside 95-105: {len(bad_sum)}")

print("\n── Composition stats (all rows) ──────────────────────────────────────────")
for col in ['Fibric_pct', 'Hemic_pct', 'Sapric_pct', 'Mineral_pct', 'Unknown_pct']:
    nonzero = (in_mn[col] > 0).sum()
    print(f"  {col}: mean={in_mn[col].mean():.1f}%  nonzero={nonzero} ({100*nonzero/len(in_mn):.1f}%)")

print("\n── Rows with any organic content (Fibric+Hemic+Sapric > 0) ──────────────")
organic_mask = (in_mn['Fibric_pct'] + in_mn['Hemic_pct'] + in_mn['Sapric_pct']) > 0
print(f"  {organic_mask.sum()} of {len(in_mn)} rows ({100*organic_mask.mean():.1f}%)")

# ── Save ───────────────────────────────────────────────────────────────────────
out_df = in_mn[KEEP_COLS]
out_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(out_df)} rows to:\n  {OUT_PATH}")
