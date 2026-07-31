"""
depth_rfe.py
=============
Recursive Feature Elimination for peat DEPTH regression.
Based on exp412 feature set and data filters.
- Dataset   : depb_features_extracted.csv
- Filter    : depb > 0 AND exp410 peat prob >= 0.5
- Features  : continuous + TC + SSURGO (exp412 feature set)
- Target    : depb (cm, raw)
- Metric    : R² (mean 5-fold CV)
- Tolerance : stop if R² drops > 0.01 below best round
- Drop      : 1 feature per round (lowest importance)

Outputs to: 05_results/rfe/depth_exp412/
  rfe_round_results.csv
  rfe_dropped_features.csv
  rfe_best_features.json
  rfe_best_feature_importances.csv
  rfe_summary.json
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
BASE      = '/scratch.global/ocon0444/peat_modeling'
MDL_DIR   = os.path.join(BASE, '03_models')
DEPTH_CSV = os.path.join(BASE, '00_data/processed/depb_features_extracted.csv')
EXP410_DIR= os.path.join(MDL_DIR, 'exp410')
EXP412_DIR= os.path.join(MDL_DIR, 'exp412')
OUT_DIR   = os.path.join(BASE, '05_results/rfe/depth_exp412')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL   = 'depb'
PROB_THRESH  = 0.5
RANDOM_STATE = 42
N_FOLDS      = 5
N_TREES      = 200
TOLERANCE    = 0.01   # stop if R² drops more than this below best

# ── FEATURE SETUP ─────────────────────────────────────────────────
CORR_JSON = os.path.join(BASE, '05_results/correlation_analysis/reduced_covariate_list.json')
with open(CORR_JSON) as f:
    corr_data = json.load(f)

EXCLUDE_ALWAYS    = ['lat','long','point_id','depb','peat_binary',
                     's2_spring_TCB','s2_spring_TCG','s2_spring_TCW',
                     's2_summer_TCB','s2_summer_TCG','s2_summer_TCW']
PEAT_MAP_EXACT    = ['gNATSGO_MN_26915','npc_peatland_indicator_10m']
PEAT_MAP_PREFIXES = ['histosols_10m_','MN_ANY_organic_component_']
SSURGO_PREFIX     = 'MN_organic_soils_classified_'
ONEHOT_PREFIXES   = ['quaternary_geology_','pennockLandformClass_','geomorphons_']
NWI_PREFIXES      = ['mn_nwi_']
TC_COLS = ['tc_spring_TCB','tc_spring_TCG','tc_spring_TCW',
           'tc_summer_TCB','tc_summer_TCG','tc_summer_TCW',
           'tc_fall_TCB','tc_fall_TCG','tc_fall_TCW']

def is_excluded(c):
    if c in EXCLUDE_ALWAYS: return True
    if c in PEAT_MAP_EXACT: return True
    if any(c.startswith(p) for p in PEAT_MAP_PREFIXES): return True
    if any(c.startswith(p) for p in ONEHOT_PREFIXES):   return True
    if any(c.startswith(p) for p in NWI_PREFIXES):      return True
    return False

# ── LOAD DATA ─────────────────────────────────────────────────────
print("Loading data...")
df_all = pd.read_csv(DEPTH_CSV, low_memory=False)

if 'mn_nwi_binary' not in df_all.columns:
    df_all['mn_nwi_binary'] = (
        (df_all['mn_nwi_cowardin_1']==1)|(df_all['mn_nwi_cowardin_2']==1)
    ).astype(int)
if 'mn_nwi_merged_1_2' not in df_all.columns:
    df_all['mn_nwi_merged_1_2'] = df_all['mn_nwi_binary']

# exp410 mask
print("Applying exp410 mask...")
with open(os.path.join(EXP410_DIR,'feature_list.json')) as f:
    exp410_feats = json.load(f)['features']
exp410_models = [pickle.load(open(os.path.join(EXP410_DIR,f'model_fold_{i}.pkl'),'rb'))
                 for i in range(N_FOLDS)]
exp410_cols = [c for c in exp410_feats if c in df_all.columns]
X_mask = df_all[exp410_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
peat_prob = np.stack([m.predict_proba(X_mask)[:,1] for m in exp410_models]).mean(axis=0)
df_all['exp410_peat_prob'] = peat_prob

mask = (df_all[TARGET_COL] > 0) & (df_all['exp410_peat_prob'] >= PROB_THRESH)
df   = df_all[mask].reset_index(drop=True)
print(f"  Filtered rows: {len(df):,} (depb>0 AND exp410>={PROB_THRESH})")

# Build exp412 feature set
ssurgo_cols        = [c for c in df.columns if c.startswith(SSURGO_PREFIX)]
reduced_continuous = [c for c in corr_data['kept_continuous']
                      if not is_excluded(c) and not c.startswith(SSURGO_PREFIX)]
all_features = [c for c in reduced_continuous + TC_COLS + ssurgo_cols if c in df.columns]
print(f"  Starting features: {len(all_features)}")

X_full = df[all_features].apply(pd.to_numeric, errors='coerce').fillna(df[all_features].median())
y      = df[TARGET_COL].values

# ── CV FUNCTION ───────────────────────────────────────────────────
def evaluate_features(X, y, feature_names):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_r2, fold_mae, fold_rmse, importances = [], [], [], []
    for tr_idx, va_idx in kf.split(X):
        rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE)
        rf.fit(X[tr_idx], y[tr_idx])
        pred = rf.predict(X[va_idx])
        fold_r2.append(r2_score(y[va_idx], pred))
        fold_mae.append(mean_absolute_error(y[va_idx], pred))
        fold_rmse.append(np.sqrt(mean_squared_error(y[va_idx], pred)))
        importances.append(rf.feature_importances_)
    mean_imp = np.mean(importances, axis=0)
    return {
        'r2':   np.mean(fold_r2),
        'mae':  np.mean(fold_mae),
        'rmse': np.mean(fold_rmse),
        'importances': dict(zip(feature_names, mean_imp)),
    }

# ── RFE LOOP ──────────────────────────────────────────────────────
print(f"\nStarting RFE ({len(all_features)} features)...")
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

    metrics = evaluate_features(X_rnd, y, current_features)
    r2, mae, rmse = metrics['r2'], metrics['mae'], metrics['rmse']
    elapsed = (time.time() - t0) / 60

    # Track best
    if r2 > best_r2:
        best_r2          = r2
        best_round       = rnd
        best_features    = list(current_features)
        best_importances = metrics['importances'].copy()

    # Find lowest importance feature to drop next
    imp_series  = pd.Series(metrics['importances']).sort_values()
    drop_feat   = imp_series.index[0]
    drop_imp    = imp_series.iloc[0]

    round_results.append({
        'round':        rnd,
        'n_features':   len(current_features),
        'r2':           round(r2, 6),
        'mae':          round(mae, 3),
        'rmse':         round(rmse, 3),
        'dropped_next': drop_feat,
        'drop_importance': round(drop_imp, 8),
        'is_best':      r2 == best_r2,
        'elapsed_min':  round(elapsed, 2),
    })

    print(f"Round {rnd:03d} | n={len(current_features):3d} | "
          f"R²={r2:.4f} | MAE={mae:.1f}cm | RMSE={rmse:.1f}cm | "
          f"drop='{drop_feat}' ({drop_imp:.5f}) | {elapsed:.1f}min")

    # Stop condition: R² dropped > tolerance below best
    if best_r2 - r2 > TOLERANCE and rnd > best_round:
        print(f"\nStopping: R² dropped {best_r2 - r2:.4f} below best (tolerance={TOLERANCE})")
        dropped_log.append({'round': rnd, 'feature': drop_feat, 'importance': drop_imp,
                            'r2_after': r2, 'stop': True})
        break

    dropped_log.append({'round': rnd, 'feature': drop_feat, 'importance': drop_imp,
                        'r2_after': r2, 'stop': False})
    current_features.remove(drop_feat)

# ── SAVE RESULTS ─────────────────────────────────────────────────
total_min = (time.time() - t_total) / 60
print(f"\nDone in {total_min:.1f} min")
print(f"Best round: {best_round} | R²={best_r2:.4f} | n_features={len(best_features)}")

pd.DataFrame(round_results).to_csv(os.path.join(OUT_DIR, 'rfe_round_results.csv'), index=False)
pd.DataFrame(dropped_log).to_csv(os.path.join(OUT_DIR, 'rfe_dropped_features.csv'), index=False)

with open(os.path.join(OUT_DIR, 'rfe_best_features.json'), 'w') as f:
    json.dump({'features': best_features, 'n_features': len(best_features),
               'best_round': best_round, 'best_r2': best_r2}, f, indent=2)

imp_df = pd.DataFrame(list(best_importances.items()),
                      columns=['feature','importance'])\
           .sort_values('importance', ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, 'rfe_best_feature_importances.csv'), index=False)

with open(os.path.join(OUT_DIR, 'rfe_summary.json'), 'w') as f:
    json.dump({
        'base_exp': 'exp412',
        'target': TARGET_COL,
        'filter': f'depb>0 AND exp410>={PROB_THRESH}',
        'n_samples': len(df),
        'starting_features': len(all_features),
        'best_features': len(best_features),
        'best_round': best_round,
        'best_r2': round(best_r2, 6),
        'total_rounds': len(round_results),
        'tolerance': TOLERANCE,
        'total_min': round(total_min, 1),
    }, f, indent=2)

print(f"\nResults saved to {OUT_DIR}")
print(f"Top 10 features at best round:")
for feat, imp in sorted(best_importances.items(), key=lambda x: -x[1])[:10]:
    print(f"  {feat:<45} {imp:.5f}")
