"""
rfe_exp421.py
=============
Recursive Feature Elimination for exp421 organic composition models.
Finds ONE shared optimal feature set across Fibric, Hemic, and Sapric targets.

Logic:
  1. Train 5-fold CV for all 3 targets with current feature set
  2. Compute mean R² across all 3 targets as the optimization metric
  3. If mean R² did NOT drop > R2_TOLERANCE vs best seen:
       - Drop the single lowest mean feature importance (averaged across all 3 models)
       - Repeat
  4. If mean R² DID drop > R2_TOLERANCE: stop.

Outputs: 05_results/rfe/exp421/
"""

import os, json, pickle, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE         = '/scratch.global/ocon0444/peat_modeling'
CSV_PATH     = os.path.join(BASE, '00_data/processed/organic_composition_features_extracted.csv')
FEAT_JSON    = os.path.join(BASE, '03_models/exp421/feature_list.json')
EXP410_DIR   = os.path.join(BASE, '03_models/exp410')
OUT_DIR      = os.path.join(BASE, '05_results/rfe/exp421')
os.makedirs(OUT_DIR, exist_ok=True)

TARGETS      = ['Fibric_pct', 'Hemic_pct', 'Sapric_pct']
PROB_THRESH  = 0.33
RANDOM_STATE = 42
N_FOLDS      = 5
N_TREES      = 200
R2_TOLERANCE = 0.01   # stop if mean R² drops more than this vs best

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
print("Loading data...")
df_all = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Shape: {df_all.shape}")

# Derive NWI columns for exp410 mask
df_all['mn_nwi_binary'] = (
    (df_all['mn_nwi_cowardin_10m'] == 1) | (df_all['mn_nwi_cowardin_10m'] == 2)
).astype(int)
df_all['mn_nwi_cowardin_0'] = (df_all['mn_nwi_cowardin_10m'] == 0).astype(int)

# Load exp421 starting feature list
with open(FEAT_JSON) as f:
    feat_data = json.load(f)
feature_cols = feat_data['features'].copy()
print(f"  Starting features: {len(feature_cols)}")

# ── APPLY EXP410 MASK ──────────────────────────────────────────────────────────
print("\nApplying exp410 mask...")
with open(os.path.join(EXP410_DIR, 'feature_list.json')) as f:
    exp410_feats = json.load(f)['features']
exp410_models = [
    pickle.load(open(os.path.join(EXP410_DIR, f'model_fold_{i}.pkl'), 'rb'))
    for i in range(N_FOLDS)
]
exp410_cols = [c for c in exp410_feats if c in df_all.columns]
X_mask = df_all[exp410_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
peat_prob = np.stack([m.predict_proba(X_mask)[:, 1] for m in exp410_models]).mean(axis=0)
df_all['exp410_peat_prob'] = peat_prob

df = df_all[df_all['exp410_peat_prob'] >= PROB_THRESH].copy().reset_index(drop=True)
df = df.dropna(subset=TARGETS).reset_index(drop=True)
df = df[df['minnesota_dem_10m'].notna()].reset_index(drop=True)
print(f"  Training rows after mask: {len(df):,}")

weights = df['coverage_weight'].values

# ── RFE FUNCTION ───────────────────────────────────────────────────────────────
def run_cv(df, feature_cols, weights):
    """Run 5-fold CV for all 3 targets. Returns mean R² across targets and
    mean feature importances averaged across targets and folds."""
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_r2       = {t: [] for t in TARGETS}
    all_imp      = {t: np.zeros(len(feature_cols)) for t in TARGETS}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        for target in TARGETS:
            y = df[target].values
            rf = RandomForestRegressor(
                n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE
            )
            rf.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=weights[tr_idx])
            pred = np.clip(rf.predict(X.iloc[va_idx]), 0, 100)
            r2 = r2_score(y[va_idx], pred)
            all_r2[target].append(r2)
            all_imp[target] += rf.feature_importances_

    # Average importances across folds
    for t in TARGETS:
        all_imp[t] /= N_FOLDS

    # Mean importance across all 3 targets — used to decide what to drop
    mean_imp = np.mean([all_imp[t] for t in TARGETS], axis=0)

    mean_r2_per_target = {t: np.mean(all_r2[t]) for t in TARGETS}
    overall_mean_r2    = np.mean(list(mean_r2_per_target.values()))

    return overall_mean_r2, mean_r2_per_target, mean_imp

# ── RFE LOOP ───────────────────────────────────────────────────────────────────
print("\nStarting RFE...")
print(f"  R² tolerance: {R2_TOLERANCE}")
print(f"  N trees: {N_TREES}  |  N folds: {N_FOLDS}")

round_results = []
best_r2       = -np.inf
best_features = feature_cols.copy()
best_round    = 0
current_feats = feature_cols.copy()
rnd           = 0

while len(current_feats) > 5:
    t0 = time.time()
    mean_r2, r2_per_target, mean_imp = run_cv(df, current_feats, weights)
    elapsed = time.time() - t0

    # Lowest importance feature to potentially drop
    drop_idx  = int(np.argmin(mean_imp))
    drop_feat = current_feats[drop_idx]

    result = {
        'round':          rnd,
        'n_features':     len(current_feats),
        'mean_r2':        round(mean_r2, 6),
        'fibric_r2':      round(r2_per_target['Fibric_pct'], 6),
        'hemic_r2':       round(r2_per_target['Hemic_pct'], 6),
        'sapric_r2':      round(r2_per_target['Sapric_pct'], 6),
        'dropped_feature': drop_feat,
        'min_importance': round(float(mean_imp[drop_idx]), 8),
        'elapsed_s':      round(elapsed, 1),
    }
    round_results.append(result)

    print(f"  Round {rnd:>3} | n={len(current_feats):>3} | "
          f"mean_R²={mean_r2:.4f} "
          f"(F={r2_per_target['Fibric_pct']:.3f} "
          f"H={r2_per_target['Hemic_pct']:.3f} "
          f"S={r2_per_target['Sapric_pct']:.3f}) | "
          f"drop='{drop_feat}' | {elapsed:.0f}s")

    # Save round results incrementally
    pd.DataFrame(round_results).to_csv(
        os.path.join(OUT_DIR, 'rfe_round_results.csv'), index=False)

    # Update best
    if mean_r2 > best_r2:
        best_r2       = mean_r2
        best_features = current_feats.copy()
        best_round    = rnd

    # Stop if R² dropped too much
    if best_r2 - mean_r2 > R2_TOLERANCE:
        print(f"\n  STOP: R² dropped {best_r2 - mean_r2:.4f} > {R2_TOLERANCE} tolerance")
        print(f"  Best was round {best_round} with {len(best_features)} features, R²={best_r2:.4f}")
        break

    # Drop lowest importance feature and continue
    current_feats = [f for f in current_feats if f != drop_feat]
    rnd += 1

# ── SAVE RESULTS ───────────────────────────────────────────────────────────────
print(f"\nSaving results to {OUT_DIR}...")

# Best features
best_feat_data = {
    'exp_id':      'exp421',
    'best_round':  best_round,
    'best_mean_r2': round(best_r2, 6),
    'n_features':  len(best_features),
    'features':    best_features,
    'tolerance':   R2_TOLERANCE,
}
with open(os.path.join(OUT_DIR, 'rfe_best_features.json'), 'w') as f:
    json.dump(best_feat_data, f, indent=2)

# Round results
round_df = pd.DataFrame(round_results)
round_df.to_csv(os.path.join(OUT_DIR, 'rfe_round_results.csv'), index=False)

# Summary
summary = {
    'start_features':   len(feature_cols),
    'best_n_features':  len(best_features),
    'features_dropped': len(feature_cols) - len(best_features),
    'best_round':       best_round,
    'best_mean_r2':     round(best_r2, 6),
    'tolerance':        R2_TOLERANCE,
    'total_rounds':     rnd,
}
with open(os.path.join(OUT_DIR, 'rfe_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Feature importances at best round
best_mean_r2, best_r2_per_target, best_imp = run_cv(df, best_features, weights)
imp_df = pd.DataFrame({
    'feature':    best_features,
    'importance': best_imp,
}).sort_values('importance', ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, 'rfe_best_feature_importances.csv'), index=False)

print(f"\n── RFE Summary ──────────────────────────────────────────────")
print(f"  Start features : {len(feature_cols)}")
print(f"  Best features  : {len(best_features)} (round {best_round})")
print(f"  Features dropped: {len(feature_cols) - len(best_features)}")
print(f"  Best mean R²   : {best_r2:.4f}")
print(f"  Per-target R² at best:")
for t, r in best_r2_per_target.items():
    print(f"    {t}: {r:.4f}")
print(f"\nTop 20 features at best round:")
print(imp_df.head(20).to_string(index=False))
print(f"\nResults saved to: {OUT_DIR}")
