#!/usr/bin/env python3
"""
exp424 — Carbon % Regression (horizon level, peat inventory)
Single target: carbon_pct
No exp410 mask — using all peat inventory horizons with valid data
Mid-depth included as covariate
"""
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

BASE       = '/scratch.global/ocon0444/peat_modeling'
MDL_DIR    = os.path.join(BASE, '03_models')
RES_DIR    = os.path.join(BASE, '05_results')
CSV_PATH   = os.path.join(BASE, '00_data/processed/carbon_features_extracted.csv')
CATALOG    = os.path.join(BASE, 'MODEL_CATALOG.csv')

TARGET       = 'carbon_pct'
RANDOM_STATE = 42
N_FOLDS      = 5
N_TREES      = 200
EXP_ID       = 'exp424'
EXP_NAME     = 'Carbon % Regression - horizon level - peat inventory'
NOTES        = ('RF regression on horizon-level carbon%. '
                '714 peat inventory horizons. No exp410 mask. '
                'Mid-depth included as covariate.')

EXCLUDE = [
    'point_id','source','DNR Peat Inventory ID #','Depthnum',
    'Top Depth (cm)','Bottom Depth (cm)',
    'Classification','Degree of Decomposition',
    'carbon_pct','carbon_source','bulk_density_gcc','lat','lon',
    's2_spring_TCB','s2_spring_TCG','s2_spring_TCW',
    's2_summer_TCB','s2_summer_TCG','s2_summer_TCW',
    'dist_from_waterbody_edge_10m',
    'gNATSGO_MN_26915','npc_peatland_indicator_10m',
    'histosols_10m_snapped','MN_organic_soils_classified_FIXED_snapped',
    'MN_ANY_organic_component_snapped',
]
ONEHOT_PREFIXES = ['quaternary_geology_','pennockLandformClass_','geomorphons_']

os.makedirs(os.path.join(MDL_DIR, EXP_ID), exist_ok=True)
os.makedirs(os.path.join(RES_DIR, EXP_ID), exist_ok=True)

print("Loading CSV...")
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df.dropna(subset=[TARGET]).reset_index(drop=True)
df = df[df['minnesota_dem_10m'].notna()].reset_index(drop=True)
print(f"  Training rows: {len(df):,}")

exclude_set = set(EXCLUDE)
feature_cols = [c for c in df.columns
                if c not in exclude_set
                and not any(c.startswith(p) for p in ONEHOT_PREFIXES)]

X = df[feature_cols].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())
y = df[TARGET].values

print(f"  Features: {len(feature_cols)}")
print(f"  carbon_pct stats: mean={y.mean():.2f}  std={y.std():.2f}  min={y.min():.1f}  max={y.max():.1f}")

print(f"\nTraining {N_FOLDS}-fold CV...")
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_results, models, y_pred_oof = [], [], np.zeros(len(y))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X.iloc[tr_idx], y[tr_idx])
    pred = rf.predict(X.iloc[va_idx])
    y_pred_oof[va_idx] = pred
    models.append(rf)
    r2   = r2_score(y[va_idx], pred)
    mae  = mean_absolute_error(y[va_idx], pred)
    rmse = np.sqrt(mean_squared_error(y[va_idx], pred))
    fold_results.append({'fold':fold,'r2':r2,'mae':mae,'rmse':rmse})
    print(f"  Fold {fold}  R²={r2:.4f}  MAE={mae:.2f}%  RMSE={rmse:.2f}%")

fold_df = pd.DataFrame(fold_results)
mean_r2   = fold_df['r2'].mean()
mean_mae  = fold_df['mae'].mean()
mean_rmse = fold_df['rmse'].mean()
print(f"\n  Mean R²  : {mean_r2:.4f} +/- {fold_df['r2'].std():.4f}")
print(f"  Mean MAE : {mean_mae:.2f}%")
print(f"  Mean RMSE: {mean_rmse:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(y, y_pred_oof, alpha=0.5, s=10, color='steelblue')
lim = [y.min()-1, y.max()+1]
axes[0].plot(lim, lim, 'r--', lw=1.5, label='1:1')
axes[0].set_xlabel('Observed Carbon%'); axes[0].set_ylabel('Predicted Carbon%')
axes[0].set_title(f'exp424 — Carbon% OOF\nR²={mean_r2:.4f}  MAE={mean_mae:.2f}%  RMSE={mean_rmse:.2f}%\nn={len(y):,}')
axes[0].legend(); axes[0].grid(alpha=0.3)

imp = np.mean([m.feature_importances_ for m in models], axis=0)
imp_df = pd.DataFrame({'feature':feature_cols,'importance':imp}).sort_values('importance',ascending=False).head(25)
imp_df.to_csv(os.path.join(RES_DIR,EXP_ID,'feature_importance.csv'), index=False)
axes[1].barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='coral', alpha=0.8)
axes[1].set_title('Top 25 Feature Importance')
axes[1].set_xlabel('Mean Importance')
axes[1].grid(alpha=0.3, axis='x')
axes[1].tick_params(axis='y', labelsize=8)
plt.suptitle('exp424 — Carbon % Regression', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR,EXP_ID,'results.png'), dpi=150, bbox_inches='tight')
print("Saved results.png")

model_dir = os.path.join(MDL_DIR, EXP_ID)
for i, m in enumerate(models):
    pickle.dump(m, open(os.path.join(model_dir,f'model_fold_{i}.pkl'),'wb'))
fold_df.to_csv(os.path.join(model_dir,'cv_results.csv'), index=False)
json.dump({
    'exp_id': EXP_ID, 'features': feature_cols, 'n_features': len(feature_cols),
    'target': TARGET, 'mask': 'none', 'n_samples': len(df),
    'mean_r2': round(mean_r2,4), 'mean_mae': round(mean_mae,2), 'mean_rmse': round(mean_rmse,2),
}, open(os.path.join(model_dir,'feature_list.json'),'w'), indent=2)

catalog = pd.read_csv(CATALOG)
catalog = catalog[catalog['exp_id'] != EXP_ID]
catalog = pd.concat([catalog, pd.DataFrame([{
    'exp_id': EXP_ID, 'exp_name': EXP_NAME, 'status': 'active',
    'task': 'carbon_regression', 'dataset': 'carbon_features_extracted.csv',
    'n_samples': len(df), 'n_features': len(feature_cols), 'n_trees': N_TREES,
    'peat_maps_excluded': True, 'mean_r2': round(mean_r2,4), 'mean_rmse': round(mean_rmse,2),
    'date': str(date.today()), 'notes': NOTES, 'model_dir': f'03_models/{EXP_ID}',
    'categoricals': 'none', 'mean_auc': '', 'mean_acc': '', 'mean_f1': '',
    'mean_avg_prec': '', 'depth_bins': '', 'output_type': 'carbon_pct',
    'output_values': '0-100%', 'nwi_included': False, 'nwi_classes': 'none', 'notebook': '',
}])], ignore_index=True)
catalog.to_csv(CATALOG, index=False)
print(f"\n{EXP_ID} saved. Catalog updated.")
