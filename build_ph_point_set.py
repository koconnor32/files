"""
build_ph_point_set.py
=====================
Builds a pH point dataset for spatial modeling by:
  1. Extracting depth-weighted average pH (0-20cm) from peat inventory (Labdnr)
  2. Extracting depth-weighted average pH (0-20cm) from NASIS organic pedons
  3. Combining, clipping to MN bbox, saving to point_data folder

Output: /scratch.global/ocon0444/peat_modeling/00_data/point_data/ph_0_20cm_combined.csv

Depth-weighted average logic:
  - Clip each horizon to the 0-20cm window
  - Weight pH by cm of overlap within window
  - weighted_pH = sum(pH * overlap_cm) / sum(overlap_cm)
  - coverage_weight = sum(overlap_cm) / 20  (capped at 1.0)
"""

import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_PI   = '/scratch.global/ocon0444/peat_inventory/peat_inventory_tables'
NASIS_CSV = '/scratch.global/ocon0444/NASIS_points/horizon_composition_0_50cm_ORGANIC_ONLY.csv'
NASIS_DB  = '/scratch.global/ocon0444/NASIS_points/mn-nasis-export.sqlite'
OUT_PATH  = '/scratch.global/ocon0444/peat_modeling/00_data/point_data/ph_0_20cm_combined.csv'

DEPTH_MIN = 0
DEPTH_MAX = 20
MN_LAT_MIN, MN_LAT_MAX =  43.5,  49.4
MN_LON_MIN, MN_LON_MAX = -97.25, -89.45

def weighted_ph(group, top_col, bot_col, ph_col):
    """
    Compute depth-weighted average pH across 0-20cm window.
    Returns (weighted_ph, total_overlap_cm).
    """
    overlap_cms = []
    ph_vals     = []
    for _, row in group.iterrows():
        top = row[top_col]
        bot = row[bot_col]
        ph  = row[ph_col]
        if pd.isna(top) or pd.isna(bot) or pd.isna(ph):
            continue
        # Clip to 0-20cm window
        clipped_top = max(top, DEPTH_MIN)
        clipped_bot = min(bot, DEPTH_MAX)
        overlap = clipped_bot - clipped_top
        if overlap > 0:
            overlap_cms.append(overlap)
            ph_vals.append(ph)
    if not overlap_cms:
        return np.nan, 0.0
    total_overlap = sum(overlap_cms)
    w_ph = sum(p * o for p, o in zip(ph_vals, overlap_cms)) / total_overlap
    return round(w_ph, 3), round(total_overlap, 1)

# ══════════════════════════════════════════════════════════════════════════════
# 1. PEAT INVENTORY
# ══════════════════════════════════════════════════════════════════════════════
print("Processing peat inventory...")
labdnr = pd.read_csv(f'{BASE_PI}/Labdnr.csv')
locate = pd.read_csv(f'{BASE_PI}/Locate.csv')

# Keep only horizons that overlap with 0-20cm window
# Overlap condition: top < 20 AND bottom > 0
pi_overlap = labdnr[
    labdnr['D_PH H2O'].notna() &
    labdnr['D_Top Depth (cm)'].notna() &
    labdnr['D_Bottom Depth (cm)'].notna() &
    (labdnr['D_Top Depth (cm)'] < DEPTH_MAX) &
    (labdnr['D_Bottom Depth (cm)'] > DEPTH_MIN)
].copy()

print(f"  Peat inventory horizons overlapping 0-20cm with pH: {len(pi_overlap)}")
print(f"  Unique sites: {pi_overlap['DNR Peat Inventory ID #'].nunique()}")

# Depth-weighted average per site
pi_rows = []
for site_id, grp in pi_overlap.groupby('DNR Peat Inventory ID #'):
    w_ph, overlap = weighted_ph(grp, 'D_Top Depth (cm)', 'D_Bottom Depth (cm)', 'D_PH H2O')
    if not np.isnan(w_ph):
        pi_rows.append({
            'point_id':        site_id,
            'source':          'peat_inventory',
            'ph_h2o':          w_ph,
            'coverage_cm':     overlap,
            'coverage_weight': min(overlap / DEPTH_MAX, 1.0),
        })

pi_df = pd.DataFrame(pi_rows)
print(f"  Sites with valid weighted pH: {len(pi_df)}")

# Add coordinates from Locate
locate_slim = locate[['DNR Peat Inventory ID #', 'UTM_E83', 'UTM_N83', 'UTM_Z83']].copy()
# Convert UTM to lat/lon using zone 15
from pyproj import Transformer
utm_rows = locate_slim[locate_slim['UTM_Z83'] == 15].copy()
utm_other = locate_slim[locate_slim['UTM_Z83'] != 15].copy()

transformer_15 = Transformer.from_crs('EPSG:26915', 'EPSG:4326', always_xy=True)
lons, lats = transformer_15.transform(utm_rows['UTM_E83'].values, utm_rows['UTM_N83'].values)
utm_rows['lat'] = lats
utm_rows['lon'] = lons

# Handle any other zones if present
if len(utm_other) > 0:
    print(f"  WARNING: {len(utm_other)} rows with non-zone-15 UTM — skipping coordinate conversion")
    utm_other['lat'] = np.nan
    utm_other['lon'] = np.nan

locate_coords = pd.concat([utm_rows, utm_other])[
    ['DNR Peat Inventory ID #', 'lat', 'lon']
]

pi_df = pi_df.merge(
    locate_coords,
    left_on='point_id',
    right_on='DNR Peat Inventory ID #',
    how='left'
).drop(columns=['DNR Peat Inventory ID #'])

print(f"  Sites with coordinates: {pi_df['lat'].notna().sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. NASIS
# ══════════════════════════════════════════════════════════════════════════════
print("\nProcessing NASIS...")

# Get organic pedon IDs from existing CSV
nasis_org = pd.read_csv(NASIS_CSV)
organic_upedons = nasis_org['upedonid'].unique().tolist()
print(f"  Organic pedons from CSV: {len(organic_upedons)}")

# Pull pH horizons for organic pedons from sqlite
conn = sqlite3.connect(NASIS_DB)

# Build query — SQLite has limits on IN clause size, chunk if needed
chunk_size = 900
all_ph_rows = []
for i in range(0, len(organic_upedons), chunk_size):
    chunk = organic_upedons[i:i+chunk_size]
    chunk_str = ','.join([f'"{u}"' for u in chunk])
    query = f'''
        SELECT p.upedonid,
               s.latstddecimaldegrees AS lat,
               s.longstddecimaldegrees AS lon,
               h.hzdept  AS top_cm,
               h.hzdepb  AS bot_cm,
               h.phfield AS ph_h2o,
               h.desgnmaster
        FROM phorizon h
        JOIN pedon p    ON h.peiidref = p.peiid
        JOIN siteobs so ON p.siteobsiidref = so.siteobsiid
        JOIN site s     ON so.siteiidref = s.siteiid
        WHERE h.phfield IS NOT NULL
          AND h.hzdept IS NOT NULL
          AND h.hzdepb IS NOT NULL
          AND h.hzdept < {DEPTH_MAX}
          AND h.hzdepb > {DEPTH_MIN}
          AND p.upedonid IN ({chunk_str})
          AND s.latstddecimaldegrees IS NOT NULL
    '''
    chunk_df = pd.read_sql(query, conn)
    all_ph_rows.append(chunk_df)

conn.close()

nasis_ph = pd.concat(all_ph_rows, ignore_index=True)
print(f"  NASIS horizons overlapping 0-20cm: {len(nasis_ph)}")
print(f"  Unique pedons: {nasis_ph['upedonid'].nunique()}")

# Depth-weighted average per pedon
nasis_rows = []
for upedonid, grp in nasis_ph.groupby('upedonid'):
    w_ph, overlap = weighted_ph(grp, 'top_cm', 'bot_cm', 'ph_h2o')
    if not np.isnan(w_ph):
        lat = grp['lat'].iloc[0]
        lon = grp['lon'].iloc[0]
        nasis_rows.append({
            'point_id':        upedonid,
            'source':          'NASIS',
            'ph_h2o':          w_ph,
            'coverage_cm':     overlap,
            'coverage_weight': min(overlap / DEPTH_MAX, 1.0),
            'lat':             lat,
            'lon':             lon,
        })

nasis_df = pd.DataFrame(nasis_rows)
print(f"  Pedons with valid weighted pH: {len(nasis_df)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. COMBINE AND CLIP TO MN
# ══════════════════════════════════════════════════════════════════════════════
print("\nCombining datasets...")
combined = pd.concat([pi_df, nasis_df], ignore_index=True)
print(f"  Combined rows before clip: {len(combined)}")

# Drop missing coords
combined = combined.dropna(subset=['lat', 'lon'])

# Clip to MN bounding box
combined = combined[
    (combined['lat'] >= MN_LAT_MIN) & (combined['lat'] <= MN_LAT_MAX) &
    (combined['lon'] >= MN_LON_MIN) & (combined['lon'] <= MN_LON_MAX)
].reset_index(drop=True)
print(f"  After MN clip: {len(combined)}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n── Source breakdown ──────────────────────────────────────────────────────")
print(combined['source'].value_counts().to_string())

print("\n── pH stats (all rows) ───────────────────────────────────────────────────")
print(combined['ph_h2o'].describe().round(3).to_string())

print("\n── pH stats by source ────────────────────────────────────────────────────")
print(combined.groupby('source')['ph_h2o'].describe().round(3).to_string())

print("\n── Coverage weight distribution ──────────────────────────────────────────")
print(combined['coverage_weight'].describe().round(3).to_string())

low_cov = (combined['coverage_weight'] < 0.5).sum()
print(f"  Rows with coverage < 50% of 0-20cm window: {low_cov}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_cols = ['point_id', 'source', 'lat', 'lon',
            'ph_h2o', 'coverage_cm', 'coverage_weight']
combined[out_cols].to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(combined)} rows to:\n  {OUT_PATH}")
