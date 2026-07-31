#!/usr/bin/env python3
"""
exp422 — Organic Composition Regression with RFE-selected features
Same as exp421 but using the 15 best features from rfe_exp421.
Adds post-hoc normalization check on OOF predictions.
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

# ── Config ─────────────────────────────────────────────────────────────────────
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
EXP_ID       = 'exp422'
EXP_NAME     = 'Organic Composition Regression - RFE 15 features - exp410>=0.33 mask'
NOTES        = ('Three-target RF regression on organic composition 0-50cm. '
                'RFE-selected 15 features from exp421 RFE. '
                'exp410>=0.33 mask. Coverage-weighted. No categoricals.')

os.makedirs(os.path.join(MDL_DIR, EXP_ID), exist_ok=True)
os.makedirs(os.path.join(RES_DIR, EXP_ID), exist_ok=True)

# ── Load RFE feature list ──────────────────────────────────────────────────────
with open(os.path.join(RFE_DIR, 'rfe_best_features.json')) as f:
    rfe_data = json.load(f)
feature_cols = rfe_data['features']
print(f'RFE features: {len(feature_cols)} (best round {rfe_data["best_round"]}, R²={rfe_data["best_mean_r2"]:.4f})')
print(f'Features: {feature_cols}')

# ── Load data ──────────────────────────────────────────────────────────────────
print('\nLoading CSV...')
df_all = pd.read_csv(CSV_PATH, low_memory=False)
print(f'  Shape: {df_all.shape}')

# Derive NWI columns for exp410 mask
df_all['mn_nwi_binary'] = (
    (df_all['mn_nwi_cowardin_10m'] == 1) | (df_all['mn_nwi_cowardin_10m'] == 2)
).astype(int)
df_all['mn_nwi_cowardin_0'] = (df_all['mn_nwi_cowardin_10m'] == 0).astype(int)

# ── Apply exp410 mask ──────────────────────────────────────────────────────────
print('\nApplying exp410 mask...')
with open(os.path.join(EXP410_DIR, 'feature_list.json')) as f:
    exp410_feats = json.load(f)['features']
exp410_models = [
    pickle.load(open(os.path.join(EXP410_DIR, f'model_fold_{i}.pkl'), 'rb'))
    for i in range(N_FOLDS)
]
exp410_cols = [c for c in exp410_feats if c in df_all.columns]
missing_410 = [c for c in exp410_feats if c not in df_all.columns]
if missing_410:
    print(f'  WARNING: {len(missing_410)} exp410 features missing: {missing_410}')
else:
    print(f'  All {len(exp410_cols)} exp410 features present.')

X_mask = df_all[exp410_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
peat_prob = np.stack([m.predict_proba(X_mask)[:, 1] for m in exp410_models]).mean(axis=0)
df_all['exp410_peat_prob'] = peat_prob
print(f'  exp410 prob >= {PROB_THRESH}: {(peat_prob >= PROB_THRESH).sum():,} / {len(peat_prob):,} rows')

# ── Filter ─────────────────────────────────────────────────────────────────────
df = df_all[df_all['exp410_peat_prob'] >= PROB_THRESH].copy().reset_index(drop=True)
df = df.dropna(subset=TARGETS).reset_index(drop=True)
df = df[df['minnesota_dem_10m'].notna()].reset_index(drop=True)
print(f'  Training rows: {len(df):,}')

# Check all RFE features present
missing_rfe = [c for c in feature_cols if c not in df.columns]
if missing_rfe:
    print(f'  WARNING: {len(missing_rfe)} RFE features missing: {missing_rfe}')
else:
    print(f'  All {len(feature_cols)} RFE features present.')

X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())
weights = df['coverage_weight'].values

print('\nTarget stats:')
for t in TARGETS:
    nonzero = (df[t] > 0).sum()
    print(f'  {t}: mean={df[t].mean():.1f}%  nonzero={nonzero} ({100*nonzero/len(df):.1f}%)')

# ── Train ──────────────────────────────────────────────────────────────────────
results = {}
for target in TARGETS:
    print(f"\n{'='*60}")
    print(f"Training: {target}")
    print('='*60)

    y = df[target].values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_results, models, y_pred_oof = [], [], np.zeros(len(y))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        rf = RandomForestRegressor(
            n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE
        )
        rf.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=weights[tr_idx])
        pred = np.clip(rf.predict(X.iloc[va_idx]), 0, 100)
        y_pred_oof[va_idx] = pred
        models.append(rf)
        r2   = r2_score(y[va_idx], pred)
        mae  = mean_absolute_error(y[va_idx], pred)
        rmse = np.sqrt(mean_squared_error(y[va_idx], pred))
        fold_results.append({'fold': fold, 'r2': r2, 'mae': mae, 'rmse': rmse})
        print(f'  Fold {fold}  R²={r2:.4f}  MAE={mae:.2f}%  RMSE={rmse:.2f}%')

    fold_df = pd.DataFrame(fold_results)
    print(f'\n  Mean R²  : {fold_df["r2"].mean():.4f} +/- {fold_df["r2"].std():.4f}')
    print(f'  Mean MAE : {fold_df["mae"].mean():.2f}%')
    print(f'  Mean RMSE: {fold_df["rmse"].mean():.2f}%')

    results[target] = {
        'models':     models,
        'fold_df':    fold_df,
        'y':          y,
        'y_pred_oof': y_pred_oof,
        'mean_r2':    round(fold_df['r2'].mean(), 4),
        'mean_mae':   round(fold_df['mae'].mean(), 2),
        'mean_rmse':  round(fold_df['rmse'].mean(), 2),
    }

# ── OOF normalization check ────────────────────────────────────────────────────
print('\n── OOF normalization check ───────────────────────────────────────────────')
oof_sum = sum(results[t]['y_pred_oof'] for t in TARGETS)
print(f'  F+H+S sum: mean={oof_sum.mean():.1f}%  std={oof_sum.std():.1f}%  '
      f'min={oof_sum.min():.1f}%  max={oof_sum.max():.1f}%')
print(f'  Pixels >100%: {(oof_sum > 100).sum()} ({100*(oof_sum>100).mean():.1f}%)')

# ── Plots ──────────────────────────────────────────────────────────────────────
colors = {'Fibric_pct': '#3498db', 'Hemic_pct': '#e67e22', 'Sapric_pct': '#8e44ad'}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, target in zip(axes, TARGETS):
    res = results[target]
    ax.scatter(res['y'], res['y_pred_oof'], alpha=0.3, s=6, color=colors[target])
    lim = max(res['y'].max(), res['y_pred_oof'].max())
    ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='1:1')
    ax.set_xlabel('Observed (%)')
    ax.set_ylabel('Predicted (%)')
    ax.set_title(f"{target}\nR²={res['mean_r2']}  MAE={res['mean_mae']}%  "
                 f"RMSE={res['mean_rmse']}%\nn={len(res['y']):,}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
plt.suptitle(f'{EXP_ID} — Organic Composition Regression (15 RFE features)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, EXP_ID, 'predicted_vs_observed.png'), dpi=150, bbox_inches='tight')
print('\nSaved predicted_vs_observed.png')

def get_color(feat):
    if feat.startswith('tc_'):    return '#e74c3c'
    if feat.startswith('s2_'):    return '#3498db'
    if feat.startswith('prism_'): return '#2ecc71'
    if feat.startswith('dist_'):  return '#e67e22'
    if feat.startswith('mn_nwi'): return '#1abc9c'
    return '#7f8c8d'

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
for ax, target in zip(axes, TARGETS):
    imp = np.mean([m.feature_importances_ for m in results[target]['models']], axis=0)
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': imp}) \
               .sort_values('importance', ascending=False)
    imp_df.to_csv(os.path.join(RES_DIR, EXP_ID,
                               f'feature_importance_{target}.csv'), index=False)
    clrs = [get_color(f) for f in imp_df['feature']]
    ax.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color=clrs[::-1])
    ax.set_title(f"{target}\nAll {len(feature_cols)} features | R²={results[target]['mean_r2']}")
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
           fontsize=9, bbox_to_anchor=(0.5, -0.02))
plt.suptitle(f'{EXP_ID} — Feature Importance (15 RFE features)', fontsize=13)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(os.path.join(RES_DIR, EXP_ID, 'feature_importance.png'), dpi=150, bbox_inches='tight')
print('Saved feature_importance.png')

# ── Save models ────────────────────────────────────────────────────────────────
model_dir = os.path.join(MDL_DIR, EXP_ID)
for target in TARGETS:
    stub = target.replace('_pct', '').lower()
    for i, m in enumerate(results[target]['models']):
        pickle.dump(m, open(os.path.join(model_dir, f'model_{stub}_fold_{i}.pkl'), 'wb'))
    results[target]['fold_df'].to_csv(
        os.path.join(model_dir, f'cv_results_{stub}.csv'), index=False)

json.dump({
    'exp_id':            EXP_ID,
    'features':          feature_cols,
    'n_features':        len(feature_cols),
    'targets':           TARGETS,
    'mask':              f'exp410_peat_prob >= {PROB_THRESH}',
    'n_samples':         len(df),
    'coverage_weighted': True,
    'rfe_source':        '05_results/rfe/exp421',
    'categoricals':      'excluded',
    'peat_maps':         'excluded',
    'results': {t: {
        'mean_r2':   results[t]['mean_r2'],
        'mean_mae':  results[t]['mean_mae'],
        'mean_rmse': results[t]['mean_rmse'],
    } for t in TARGETS},
}, open(os.path.join(model_dir, 'feature_list.json'), 'w'), indent=2)
print(f'\n{EXP_ID}: models saved to {model_dir}')

# ── Update catalog ─────────────────────────────────────────────────────────────
catalog = pd.read_csv(CATALOG)
catalog = catalog[catalog['exp_id'] != EXP_ID]
new_row = {
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
    'output_type':        'Fibric_pct / Hemic_pct / Sapric_pct',
    'output_values':      '0-100% per class',
    'mean_acc':           '',
    'mean_f1':            '',
    'mean_auc':           '',
    'mean_r2':   round(np.mean([results[t]['mean_r2']   for t in TARGETS]), 4),
    'mean_rmse': round(np.mean([results[t]['mean_rmse'] for t in TARGETS]), 2),
    'date':               str(date.today()),
    'notes':              NOTES,
    'notebook':           'exp422_organic_rfe.py',
    'model_dir':          f'03_models/{EXP_ID}',
    'categoricals':       'none',
    'mean_avg_prec':      '',
}
catalog = pd.concat([catalog, pd.DataFrame([new_row])], ignore_index=True)
catalog.to_csv(CATALOG, index=False)
print('MODEL_CATALOG updated.')
print(catalog[catalog['exp_id'] == EXP_ID]
      [['exp_id', 'n_samples', 'n_features', 'mean_r2', 'mean_rmse']].to_string(index=False))
