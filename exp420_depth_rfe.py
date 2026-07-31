"""
exp420_depth_rfe.py
====================
Recursive Feature Elimination for depth regression — exp420.

Feature set:
  - ALL continuous covariates (correlation-filtered)
  - TC bands (spring/summer/fall)
  - NWI 3-class one-hot (mn_nwi_cowardin_0, _1, _2)
  - dist_from_waterbody_edge_10m (NaN filled with 99999)
  - NO categorical one-hots (no SSURGO, no quaternary geology, etc.)

Filter : depb > 0 AND exp410 peat prob >= 0.33
Target : depb (cm)
Metric : R² (5-fold CV)
Stop   : R² drops > 0.01 below best round

Outputs: 05_results/rfe/depth_exp420/
"""

import os, json, pickle, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────
BASE       = '/scratch.global/ocon0444/peat_modeling'
MDL_DIR    = os.path.join(BASE, '03_models')
DEPTH_CSV  = os.path.join(BASE, '00_data/processed/depb_features_extracted.csv')
EXP410_DIR = os.path.join(MDL_DIR, 'exp410')
CORR_JSON  = os.path.join(BASE, '05_results/correlation_analysis/reduced_covariate_list.json')
OUT_DIR    = os.path.join(BASE, '05_results/rfe/depth_exp420')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL   = 'depb'
PROB_THRESH  = 0.33
RANDOM_STATE = 42
N_FOLDS      = 5
N_TREES      = 200
TOLERANCE    = 0.01
NAN_FILL_DIST = 99999  # fill for dist_from_waterbody_edge NaN

# ── FEATURE RULES ─────────────────────────────────────────────────
EXCLUDE_ALWAYS = [
    'lat', 'long', 'point_id', 'depb', 'peat_binary',
    's2_spring_TCB', 's2_spring_TCG', 's2_spring_TCW',
    's2_summer_TCB', 's2_summer_TCG', 's2_summer_TCW',
]
PEAT_MAP_EXACT    = ['gNATSGO_MN_26915', 'npc_peatland_indicator_10m']
PEAT_MAP_PREFIXES = ['histosols_10m_', 'MN_ANY_organic_component_']
CATEGORICAL_PREFIXES = [
    'MN_organic_soils_classified_',
    'quaternary_geology_',
    'pennockLandformClass_',
    'geomorphons_',
]
# NWI: keep the 3 one-hot columns, drop the merged binary
NWI_EXCLUDE = ['mn_nwi_merged_1_2', 'mn_nwi_binary']

TC_COLS = [
    'tc_spring_TCB', 'tc_spring_TCG', 'tc_spring_TCW',
    'tc_summer_TCB', 'tc_summer_TCG', 'tc_summer_TCW',
    'tc_fall_TCB',   'tc_fall_TCG',   'tc_fall_TCW',
]
NWI_COLS    = ['mn_nwi_cowardin_0', 'mn_nwi_cowardin_1', 'mn_nwi_cowardin_2']
NEW_COVARS  = ['dist_from_waterbody_edge_10m']

def is_excluded(c):
    if c in EXCLUDE_ALWAYS:                                   return True
    if c in PEAT_MAP_EXACT:                                   return True
    if c in NWI_EXCLUDE:                                      return True
    if any(c.startswith(p) for p in PEAT_MAP_PREFIXES):      return True
    if any(c.startswith(p) for p in CATEGORICAL_PREFIXES):   return True
    return False

# ── LOAD DATA ─────────────────────────────────────────────────────
print("Loading data...")
with open(CORR_JSON) as f:
    corr_data = json.load(f)

df_all = pd.read_csv(DEPTH_CSV, low_memory=False)
print(f"  CSV shape: {df_all.shape}")

# Build NWI one-hot columns if not present
nwi_raw_col = None
for candidate in ['mn_nwi_cowardin_10m', 'mn_nwi', 'nwi']:
    if candidate in df_all.columns:
        nwi_raw_col = candidate
        break

for val, col in enumerate(['mn_nwi_cowardin_0', 'mn_nwi_cowardin_1', 'mn_nwi_cowardin_2']):
    if col not in df_all.columns:
        if nwi_raw_col:
            df_all[col] = (df_all[nwi_raw_col] == val).astype(int)
        elif 'mn_nwi_cowardin_1' in df_all.columns and val == 0:
            # reconstruct 0 from 1 and 2
            df_all['mn_nwi_cowardin_0'] = (
                (df_all.get('mn_nwi_cowardin_1', 0) == 0) &
                (df_all.get('mn_nwi_cowardin_2', 0) == 0)
            ).astype(int)

# Create mn_nwi_binary for exp410 mask
if 'mn_nwi_binary' not in df_all.columns:
    df_all['mn_nwi_binary'] = (
        (df_all.get('mn_nwi_cowardin_1', pd.Series(0, index=df_all.index)) == 1) |
        (df_all.get('mn_nwi_cowardin_2', pd.Series(0, index=df_all.index)) == 1)
    ).astype(int)

# exp410 mask at 0.33
print("Applying exp410 mask...")
with open(os.path.join(EXP410_DIR, 'feature_list.json')) as f:
    exp410_feats = json.load(f)['features']
exp410_models = [
    pickle.load(open(os.path.join(EXP410_DIR, f'model_fold_{i}.pkl'), 'rb'))
    for i in range(N_FOLDS)
]
exp410_cols = [c for c in exp410_feats if c in df_all.columns]
X_mask = df_all[exp410_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
peat_prob = np.stack([m.predict_proba(X_mask)[:,1] for m in exp410_models]).mean(axis=0)
df_all['exp410_peat_prob'] = peat_prob

mask = (df_all[TARGET_COL] > 0) & (df_all['exp410_peat_prob'] >= PROB_THRESH)
df   = df_all[mask].reset_index(drop=True)
print(f"  Rows after filter: {len(df):,}")

# Build feature list
reduced_continuous = [
    c for c in corr_data['kept_continuous']
    if not is_excluded(c) and c in df.columns
]
nwi_cols_present = [c for c in NWI_COLS if c in df.columns]
tc_cols_present  = [c for c in TC_COLS   if c in df.columns]
new_cols_present = [c for c in NEW_COVARS if c in df.columns]

missing_new = [c for c in NEW_COVARS if c not in df.columns]
if missing_new:
    print(f"  WARNING: {missing_new} not in CSV — run extract_waterbody_edge.py first")

all_features = list(dict.fromkeys(
    reduced_continuous + tc_cols_present + nwi_cols_present + new_cols_present
))

print(f"\nFeature breakdown:")
print(f"  Continuous (corr-filtered) : {len(reduced_continuous)}")
print(f"  TC bands                   : {len(tc_cols_present)}")
print(f"  NWI 3-class one-hot        : {len(nwi_cols_present)}")
print(f"  New covariates             : {len(new_cols_present)}")
print(f"  Total                      : {len(all_features)}")

# Prep X — NaN fill strategy:
#   dist_from_waterbody_edge_10m -> 99999 (no waterbody nearby)
#   all other columns            -> column median
X_full = df[all_features].apply(pd.to_numeric, errors='coerce')
for col in all_features:
    if col in NEW_COVARS:
        X_full[col] = X_full[col].fillna(NAN_FILL_DIST)
    else:
        X_full[col] = X_full[col].fillna(X_full[col].median())

y = df[TARGET_COL].values
print(f"\nX shape: {X_full.shape}  |  y stats: mean={y.mean():.1f}  max={y.max():.1f}")

# ── CV FUNCTION ───────────────────────────────────────────────────
def evaluate_features(X_arr, y, feature_names):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r2s, maes, rmses, imps = [], [], [], []
    for tr_idx, va_idx in kf.split(X_arr):
        rf = RandomForestRegressor(
            n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE)
        rf.fit(X_arr[tr_idx], y[tr_idx])
        pred = rf.predict(X_arr[va_idx])
        r2s.append(r2_score(y[va_idx], pred))
        maes.append(mean_absolute_error(y[va_idx], pred))
        rmses.append(np.sqrt(mean_squared_error(y[va_idx], pred)))
        imps.append(rf.feature_importances_)
    return {
        'r2':          np.mean(r2s),
        'mae':         np.mean(maes),
        'rmse':        np.mean(rmses),
        'importances': dict(zip(feature_names, np.mean(imps, axis=0))),
    }

# ── RFE LOOP ──────────────────────────────────────────────────────
print(f"\nStarting RFE ({len(all_features)} features, tolerance={TOLERANCE})...")
current_features = list(all_features)
round_results    = []
dropped_log      = []
best_r2          = -np.inf
best_round       = 0
best_features    = list(current_features)
best_importances = {}
t_total          = time.time()

for rnd in range(len(all_features) - 1):
    t0 = time.time()
    X_rnd = X_full[current_features].values

    m = evaluate_features(X_rnd, y, current_features)
    r2, mae, rmse = m['r2'], m['mae'], m['rmse']
    elapsed = (time.time() - t0) / 60

    if r2 > best_r2:
        best_r2          = r2
        best_round       = rnd
        best_features    = list(current_features)
        best_importances = m['importances'].copy()

    imp_series = pd.Series(m['importances']).sort_values()
    drop_feat  = imp_series.index[0]
    drop_imp   = imp_series.iloc[0]

    round_results.append({
        'round':           rnd,
        'n_features':      len(current_features),
        'r2':              round(r2, 6),
        'mae':             round(mae, 3),
        'rmse':            round(rmse, 3),
        'dropped_next':    drop_feat,
        'drop_importance': round(drop_imp, 8),
        'is_best':         r2 == best_r2,
        'elapsed_min':     round(elapsed, 2),
    })

    print(f"Round {rnd:03d} | n={len(current_features):3d} | "
          f"R²={r2:.4f} | MAE={mae:.1f}cm | RMSE={rmse:.1f}cm | "
          f"drop='{drop_feat}' ({drop_imp:.5f}) | {elapsed:.1f}min")

    # Stop if R² dropped too far below best
    if best_r2 - r2 > TOLERANCE and rnd > best_round:
        print(f"\nStopping: R² dropped {best_r2 - r2:.4f} below best (tolerance={TOLERANCE})")
        dropped_log.append({'round': rnd, 'feature': drop_feat,
                            'importance': drop_imp, 'stop': True})
        break

    dropped_log.append({'round': rnd, 'feature': drop_feat,
                        'importance': drop_imp, 'stop': False})
    current_features.remove(drop_feat)

# ── SAVE RESULTS ─────────────────────────────────────────────────
total_min = (time.time() - t_total) / 60
print(f"\nDone in {total_min:.1f} min")
print(f"Best round: {best_round} | R²={best_r2:.4f} | n_features={len(best_features)}")

pd.DataFrame(round_results).to_csv(
    os.path.join(OUT_DIR, 'rfe_round_results.csv'), index=False)
pd.DataFrame(dropped_log).to_csv(
    os.path.join(OUT_DIR, 'rfe_dropped_features.csv'), index=False)

with open(os.path.join(OUT_DIR, 'rfe_best_features.json'), 'w') as f:
    json.dump({
        'features':      best_features,
        'n_features':    len(best_features),
        'best_round':    best_round,
        'best_r2':       round(best_r2, 6),
        'nan_fill_dist': NAN_FILL_DIST,
        'new_covars':    NEW_COVARS,
        'nwi_type':      '3class',
        'depth_filter':  f'depb>0 AND exp410>={PROB_THRESH}',
    }, f, indent=2)

pd.DataFrame(
    sorted(best_importances.items(), key=lambda x: -x[1]),
    columns=['feature', 'importance']
).to_csv(os.path.join(OUT_DIR, 'rfe_best_feature_importances.csv'), index=False)

with open(os.path.join(OUT_DIR, 'rfe_summary.json'), 'w') as f:
    json.dump({
        'exp_id':             'exp420',
        'target':             TARGET_COL,
        'filter':             f'depb>0 AND exp410>={PROB_THRESH}',
        'n_samples':          len(df),
        'starting_features':  len(all_features),
        'best_features':      len(best_features),
        'best_round':         best_round,
        'best_r2':            round(best_r2, 6),
        'total_rounds':       len(round_results),
        'tolerance':          TOLERANCE,
        'nan_fill_dist':      NAN_FILL_DIST,
        'total_min':          round(total_min, 1),
    }, f, indent=2)

print(f"\nResults saved to {OUT_DIR}")
print(f"\nTop 10 features at best round:")
for feat, imp in sorted(best_importances.items(), key=lambda x: -x[1])[:10]:
    print(f"  {feat:<50} {imp:.5f}")
