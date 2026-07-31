#!/usr/bin/env python3
"""
exp426_compositional_rf.py
===========================
Compositional RF for Fibric/Hemic/Sapric % using iterative residual fitting.
Enforces Fibric + Hemic + Sapric = 100% at every prediction point.

Iterative residual approach:
  Step 1: Train RF on Fibric_pct directly
  Step 2: Train RF on Hemic_pct / (Hemic_pct + Sapric_pct) — fraction of remainder
  Step 3: Sapric = remainder * (1 - hemic_fraction)
          Hemic  = remainder * hemic_fraction
          remainder = 100 - Fibric_predicted

Uses exp410 >= 0.33 mask, coverage-weighted, RFE 15 features from exp422.
"""

import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE       = '/scratch.global/ocon0444/peat_modeling'
MDL_DIR    = os.path.join(BASE, '03_models')
RES_DIR    = os.path.join(BASE, '05_results')
CSV_PATH   = os.path.join(BASE, '00_data/processed/organic_composition_features_extracted.csv')
EXP410_DIR = os.path.join(MDL_DIR, 'exp410')
RFE_DIR    = os.path.join(BASE, '05_results/rfe/exp421')
CATALOG    = os.path.join(BASE, 'MODEL_CATALOG.csv')

TARGETS      = ['Fibric_pct', 'Hemic_pct', 'Sapric_pct']
PROB_THRESH  = 0.33
RANDOM_STATE = 42
N_FOLDS      = 5
N_TREES      = 200
EXP_ID       = 'exp426'
EXP_NAME     = 'Compositional RF - Iterative Residual - Fibric/Hemic/Sapric sum=100%'
NOTES        = ('Iterative residual compositional RF. Fibric predicted first, '
                'then Hemic fraction of remainder, Sapric gets rest. '
                'Fibric+Hemic+Sapric=100% by construction. '
                'RFE 15 features, exp410>=0.33 mask, coverage-weighted.')

os.makedirs(os.path.join(MDL_DIR, EXP_ID), exist_ok=True)
os.makedirs(os.path.join(RES_DIR, EXP_ID), exist_ok=True)

# ── LOAD RFE FEATURES ─────────────────────────────────────────────────────────
with open(os.path.join(RFE_DIR, 'rfe_best_features.json')) as f:
    rfe_data = json.load(f)
feature_cols = rfe_data['features']
print(f'RFE features: {len(feature_cols)}')
print(f'Features: {feature_cols}')

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
print('\nLoading CSV...')
df_all = pd.read_csv(CSV_PATH, low_memory=False)
print(f'  Shape: {df_all.shape}')

# Derive NWI columns for exp410 mask
df_all['mn_nwi_binary'] = (
    (df_all['mn_nwi_cowardin_10m'] == 1) | (df_all['mn_nwi_cowardin_10m'] == 2)
).astype(int)
df_all['mn_nwi_cowardin_0'] = (df_all['mn_nwi_cowardin_10m'] == 0).astype(int)

# ── EXP410 MASK ────────────────────────────────────────────────────────────────
print('Applying exp410 mask...')
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
print(f'  exp410 >= {PROB_THRESH}: {(peat_prob >= PROB_THRESH).sum():,} / {len(peat_prob):,}')

df = df_all[df_all['exp410_peat_prob'] >= PROB_THRESH].copy().reset_index(drop=True)
df = df.dropna(subset=TARGETS).reset_index(drop=True)
df = df[df['minnesota_dem_10m'].notna()].reset_index(drop=True)
print(f'  Training rows: {len(df):,}')

# Check features
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    print(f'  WARNING: missing features: {missing}')

X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())
weights = df['coverage_weight'].values

# ── PREPARE COMPOSITIONAL TARGETS ─────────────────────────────────────────────
# Normalize observed values so Fibric+Hemic+Sapric = 100% in training data too
organic_sum = df[TARGETS].sum(axis=1)
# Where sum is 0 assign equal shares
zero_mask = organic_sum == 0
df.loc[zero_mask, 'Fibric_pct'] = 33.33
df.loc[zero_mask, 'Hemic_pct']  = 33.33
df.loc[zero_mask, 'Sapric_pct'] = 33.34
organic_sum = df[TARGETS].sum(axis=1)

fibric_norm = (df['Fibric_pct'] / organic_sum * 100).values
hemic_norm  = (df['Hemic_pct']  / organic_sum * 100).values
sapric_norm = (df['Sapric_pct'] / organic_sum * 100).values

# Hemic fraction of (Hemic + Sapric) remainder
hemic_sapric_sum = hemic_norm + sapric_norm
# Avoid division by zero
hemic_frac = np.where(
    hemic_sapric_sum > 0,
    hemic_norm / hemic_sapric_sum,
    0.5  # equal split if both zero
)

print(f'\nNormalized training targets:')
print(f'  Fibric : mean={fibric_norm.mean():.1f}%  std={fibric_norm.std():.1f}%')
print(f'  Hemic  : mean={hemic_norm.mean():.1f}%   std={hemic_norm.std():.1f}%')
print(f'  Sapric : mean={sapric_norm.mean():.1f}%  std={sapric_norm.std():.1f}%')
print(f'  Hemic fraction of remainder: mean={hemic_frac.mean():.3f}  std={hemic_frac.std():.3f}')

# ── 5-FOLD CV ─────────────────────────────────────────────────────────────────
print(f'\nTraining {N_FOLDS}-fold CV (iterative residual)...')
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

fold_results = []
models_fibric = []
models_hemic_frac = []

oof_fibric     = np.zeros(len(df))
oof_hemic      = np.zeros(len(df))
oof_sapric     = np.zeros(len(df))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):

    # ── Step 1: predict Fibric_pct ────────────────────────────────────────────
    rf_fibric = RandomForestRegressor(
        n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_fibric.fit(X.iloc[tr_idx], fibric_norm[tr_idx],
                  sample_weight=weights[tr_idx])
    pred_fibric = np.clip(rf_fibric.predict(X.iloc[va_idx]), 0, 100)

    # ── Step 2: predict Hemic fraction of remainder ───────────────────────────
    rf_hemic_frac = RandomForestRegressor(
        n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_hemic_frac.fit(X.iloc[tr_idx], hemic_frac[tr_idx],
                      sample_weight=weights[tr_idx])
    pred_hemic_frac = np.clip(rf_hemic_frac.predict(X.iloc[va_idx]), 0, 1)

    # ── Step 3: compute final predictions ────────────────────────────────────
    remainder     = np.clip(100 - pred_fibric, 0, 100)
    pred_hemic    = remainder * pred_hemic_frac
    pred_sapric   = remainder * (1 - pred_hemic_frac)

    # Verify sum = 100
    pred_sum = pred_fibric + pred_hemic + pred_sapric
    assert np.allclose(pred_sum, 100, atol=1e-6), f'Sum not 100: {pred_sum[:5]}'

    oof_fibric[va_idx]  = pred_fibric
    oof_hemic[va_idx]   = pred_hemic
    oof_sapric[va_idx]  = pred_sapric

    models_fibric.append(rf_fibric)
    models_hemic_frac.append(rf_hemic_frac)

    # Metrics
    r2_f  = r2_score(fibric_norm[va_idx],  pred_fibric)
    r2_h  = r2_score(hemic_norm[va_idx],   pred_hemic)
    r2_s  = r2_score(sapric_norm[va_idx],  pred_sapric)
    mae_f = mean_absolute_error(fibric_norm[va_idx], pred_fibric)
    mae_h = mean_absolute_error(hemic_norm[va_idx],  pred_hemic)
    mae_s = mean_absolute_error(sapric_norm[va_idx], pred_sapric)

    fold_results.append({
        'fold': fold,
        'r2_fibric': r2_f, 'r2_hemic': r2_h, 'r2_sapric': r2_s,
        'mean_r2': (r2_f + r2_h + r2_s) / 3,
        'mae_fibric': mae_f, 'mae_hemic': mae_h, 'mae_sapric': mae_s,
    })
    print(f'  Fold {fold}  Fibric R²={r2_f:.4f}  Hemic R²={r2_h:.4f}  '
          f'Sapric R²={r2_s:.4f}  Mean R²={(r2_f+r2_h+r2_s)/3:.4f}')

fold_df = pd.DataFrame(fold_results)
print(f'\n── OOF Summary ────────────────────────────────────────────')
for t, oof_pred, obs in [('Fibric', oof_fibric, fibric_norm),
                          ('Hemic',  oof_hemic,  hemic_norm),
                          ('Sapric', oof_sapric, sapric_norm)]:
    r2   = r2_score(obs, oof_pred)
    mae  = mean_absolute_error(obs, oof_pred)
    rmse = np.sqrt(mean_squared_error(obs, oof_pred))
    print(f'  {t:<8} R²={r2:.4f}  MAE={mae:.2f}%  RMSE={rmse:.2f}%')

oof_sum = oof_fibric + oof_hemic + oof_sapric
print(f'\n  OOF sum check: mean={oof_sum.mean():.2f}  '
      f'min={oof_sum.min():.2f}  max={oof_sum.max():.2f}  '
      f'(should be exactly 100.00)')

# Compare to exp422 (independent models)
print(f'\n── vs exp422 (independent models, post-hoc normalized) ────')
print(f'  exp422: Fibric R²=0.361  Hemic R²=0.352  Sapric R²=0.589  Mean=0.434')
print(f'  exp426: Fibric R²={fold_df["r2_fibric"].mean():.3f}  '
      f'Hemic R²={fold_df["r2_hemic"].mean():.3f}  '
      f'Sapric R²={fold_df["r2_sapric"].mean():.3f}  '
      f'Mean={fold_df["mean_r2"].mean():.3f}')

# ── PLOTS ──────────────────────────────────────────────────────────────────────
colors = {'Fibric': '#3498db', 'Hemic': '#e67e22', 'Sapric': '#8e44ad'}
obs_list  = [fibric_norm, hemic_norm, sapric_norm]
oof_list  = [oof_fibric,  oof_hemic,  oof_sapric]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, name, obs, pred in zip(axes, ['Fibric','Hemic','Sapric'], obs_list, oof_list):
    r2  = r2_score(obs, pred)
    mae = mean_absolute_error(obs, pred)
    ax.scatter(obs, pred, alpha=0.25, s=6, color=colors[name])
    lim = max(obs.max(), pred.max())
    ax.plot([0,lim],[0,lim],'r--',lw=1.5,label='1:1')
    ax.set_xlabel('Observed (normalized %)')
    ax.set_ylabel('Predicted (%)')
    ax.set_title(f'{name}\nR²={r2:.4f}  MAE={mae:.2f}%\nn={len(obs):,}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle(f'{EXP_ID} — Compositional RF (sum=100% by construction)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, EXP_ID, 'predicted_vs_observed.png'),
            dpi=150, bbox_inches='tight')
print('\nSaved predicted_vs_observed.png')

# Feature importance
def get_color(feat):
    if feat.startswith('tc_'):    return '#e74c3c'
    if feat.startswith('s2_'):    return '#3498db'
    if feat.startswith('prism_'): return '#2ecc71'
    if feat.startswith('dist_'):  return '#e67e22'
    return '#7f8c8d'

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, model_list, title in [
    (axes[0], models_fibric,     'Step 1: Fibric %'),
    (axes[1], models_hemic_frac, 'Step 2: Hemic fraction of remainder'),
]:
    imp = np.mean([m.feature_importances_ for m in model_list], axis=0)
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': imp})\
               .sort_values('importance', ascending=False)
    clrs = [get_color(f) for f in imp_df['feature']]
    ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color=clrs[::-1])
    r2_label = fold_df['r2_fibric'].mean() if 'Fibric' in title else fold_df['r2_hemic'].mean() if 'Hemic' in title else None
    ax.set_title(f'{title}\nR²={r2_label:.4f}' if r2_label is not None else f'{title}')
    ax.set_xlabel('Mean Importance')
    ax.grid(alpha=0.3, axis='x')
    ax.tick_params(axis='y', labelsize=8)

legend_patches = [
    mpatches.Patch(color='#e74c3c', label='Tasseled Cap'),
    mpatches.Patch(color='#3498db', label='Sentinel-2'),
    mpatches.Patch(color='#2ecc71', label='Climate'),
    mpatches.Patch(color='#e67e22', label='Distance'),
    mpatches.Patch(color='#7f8c8d', label='Terrain'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=5,
           fontsize=9, bbox_to_anchor=(0.5,-0.02))
plt.suptitle(f'{EXP_ID} — Feature Importance', fontsize=13)
plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig(os.path.join(RES_DIR, EXP_ID, 'feature_importance.png'),
            dpi=150, bbox_inches='tight')
print('Saved feature_importance.png')

# ── SAVE MODELS ────────────────────────────────────────────────────────────────
model_dir = os.path.join(MDL_DIR, EXP_ID)
for i, (mf, mh) in enumerate(zip(models_fibric, models_hemic_frac)):
    pickle.dump(mf, open(os.path.join(model_dir, f'model_fibric_fold_{i}.pkl'), 'wb'))
    pickle.dump(mh, open(os.path.join(model_dir, f'model_hemic_frac_fold_{i}.pkl'), 'wb'))

fold_df.to_csv(os.path.join(model_dir, 'cv_results.csv'), index=False)

json.dump({
    'exp_id':            EXP_ID,
    'features':          feature_cols,
    'n_features':        len(feature_cols),
    'targets':           TARGETS,
    'method':            'iterative_residual',
    'step1_target':      'Fibric_pct (normalized to sum=100)',
    'step2_target':      'Hemic / (Hemic + Sapric) fraction',
    'mask':              f'exp410_peat_prob >= {PROB_THRESH}',
    'n_samples':         len(df),
    'coverage_weighted': True,
    'rfe_source':        '05_results/rfe/exp421',
    'results': {
        'fibric_r2':    round(fold_df['r2_fibric'].mean(), 4),
        'hemic_r2':     round(fold_df['r2_hemic'].mean(), 4),
        'sapric_r2':    round(fold_df['r2_sapric'].mean(), 4),
        'mean_r2':      round(fold_df['mean_r2'].mean(), 4),
        'fibric_mae':   round(fold_df['mae_fibric'].mean(), 2),
        'hemic_mae':    round(fold_df['mae_hemic'].mean(), 2),
        'sapric_mae':   round(fold_df['mae_sapric'].mean(), 2),
    },
}, open(os.path.join(model_dir, 'feature_list.json'), 'w'), indent=2)

# ── CATALOG ────────────────────────────────────────────────────────────────────
catalog = pd.read_csv(CATALOG)
catalog = catalog[catalog['exp_id'] != EXP_ID]
catalog = pd.concat([catalog, pd.DataFrame([{
    'exp_id':             EXP_ID,
    'exp_name':           EXP_NAME,
    'status':             'active',
    'task':               'composition_regression',
    'dataset':            'organic_composition_features_extracted.csv',
    'n_samples':          len(df),
    'n_features':         len(feature_cols),
    'n_trees':            N_TREES,
    'peat_maps_excluded': True,
    'nwi_included':       False,
    'nwi_classes':        'none',
    'depth_bins':         '',
    'output_type':        'Fibric/Hemic/Sapric compositional (sum=100%)',
    'output_values':      '0-100% each, sum=100%',
    'mean_acc': '', 'mean_f1': '', 'mean_auc': '', 'mean_avg_prec': '',
    'mean_r2':   round(fold_df['mean_r2'].mean(), 4),
    'mean_rmse': round(fold_df[['mae_fibric','mae_hemic','mae_sapric']].mean().mean(), 2),
    'date':       str(date.today()),
    'notes':      NOTES,
    'notebook':   'exp426_results_viewer.ipynb',
    'model_dir':  f'03_models/{EXP_ID}',
    'categoricals': 'none',
}])], ignore_index=True)
catalog.to_csv(CATALOG, index=False)
print(f'\n{EXP_ID} saved. Catalog updated.')
print(catalog[catalog['exp_id'] == EXP_ID]
      [['exp_id','n_samples','n_features','mean_r2']].to_string(index=False))
