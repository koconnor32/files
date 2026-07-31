#!/bin/bash
#SBATCH --job-name=agc_rf_train3
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/agc_rf_train3_%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/agc_rf_train3_%j.err

echo "=============================="
echo "AGC RF Training — Fast Version"
echo "50k points, batch raster extraction"
echo "Started: $(date)"
echo "=============================="

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os, json, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

BASE_M   = '/scratch.global/ocon0444/peat_modeling'
COV_DIR  = f'{BASE_M}/00_data/covariates_10m'
PRED_DIR = f'{BASE_M}/04_predictions'
OUT_DIR  = f'{BASE_M}/00_data/processed'
MDL_DIR  = f'{BASE_M}/03_models/exp_agc'
RES_DIR  = f'{BASE_M}/05_results/exp_agc'
os.makedirs(MDL_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

SUBSAMPLE_PATH = f'{OUT_DIR}/agc_training_points_300k.csv'
EXTRACTED_PATH = f'{OUT_DIR}/agc_features_extracted3.csv'
RFE_PATH       = f'{RES_DIR}/rfe_agc.json'
MODEL_PATH     = f'{MDL_DIR}/agc_jenkins_rf_final.pkl'
FEAT_PATH      = f'{MDL_DIR}/agc_jenkins_rf_features.json'

RANDOM_STATE  = 42
N_TREES_RFE   = 100
N_TREES_FINAL = 300
N_FOLDS       = 5
TARGET_PTS    = 50000   # reduced from 300k
TARGET        = 'C_AGC_KGM2'
np.random.seed(RANDOM_STATE)

# ── STEP 1: Subsample to 50k ───────────────────────────────────────
print('='*60)
print('STEP 1: Subsample to 50k points')
print('='*60)

# Load existing 300k subsample — already stratified by species group
df_300k = pd.read_csv(SUBSAMPLE_PATH, low_memory=False)
print(f'Loaded 300k subsample: {len(df_300k):,} points')

# Further subsample to 50k — stratified by species group
sampled = []
for grp, group_df in df_300k.groupby('JENKINS_GR'):
    n_sample = max(1, int(TARGET_PTS * len(group_df) / len(df_300k)))
    n_sample = min(n_sample, len(group_df))
    sampled.append(group_df.sample(n=n_sample, random_state=RANDOM_STATE))

df_sub = pd.concat(sampled).reset_index(drop=True)
print(f'Subsampled to: {len(df_sub):,} points from {df_sub["STAND_KEY"].nunique():,} polygons')
print('Species group distribution:')
for grp, n in df_sub['JENKINS_GR'].value_counts().items():
    print(f'  {grp:<20}: {n:,}')

coords = list(zip(df_sub['x'].values, df_sub['y'].values))

# ── STEP 2: Fast batch extraction ─────────────────────────────────
# KEY FIX: open each raster ONCE, sample ALL points, close
# Previously opened raster once per point — 125x slower
print('\n' + '='*60)
print('STEP 2: Fast batch raster extraction')
print(f'Extracting {len(coords):,} points from each raster...')
print('='*60)

if os.path.exists(EXTRACTED_PATH):
    print(f'Already exists — loading {EXTRACTED_PATH}')
    feat_df = pd.read_csv(EXTRACTED_PATH, low_memory=False)
    print(f'Loaded: {feat_df.shape}')
else:
    feat_df = df_sub.copy()
    t0 = time.time()

    BAND_MAP = {
        's2_spring_B02':1,'s2_spring_B03':2,'s2_spring_B04':3,'s2_spring_B05':4,
        's2_spring_B06':5,'s2_spring_B07':6,'s2_spring_B08':7,'s2_spring_B8A':8,
        's2_spring_B11':9,'s2_spring_B12':10,'s2_spring_NDVI':11,'s2_spring_SWDI':12,
        's2_summer_B02':1,'s2_summer_B03':2,'s2_summer_B04':3,'s2_summer_B05':4,
        's2_summer_B06':5,'s2_summer_B07':6,'s2_summer_B08':7,'s2_summer_B8A':8,
        's2_summer_B11':9,'s2_summer_B12':10,'s2_summer_NDVI':11,'s2_summer_SWDI':12,
        's2_fall_B02':1,'s2_fall_B03':2,'s2_fall_B04':3,'s2_fall_B05':4,
        's2_fall_B06':5,'s2_fall_B07':6,'s2_fall_B08':7,'s2_fall_B8A':8,
        's2_fall_B11':9,'s2_fall_B12':10,'s2_fall_NDVI':11,'s2_fall_SWDI':12,
        'tc_spring_TCB':1,'tc_spring_TCG':2,'tc_spring_TCW':3,
        'tc_summer_TCB':1,'tc_summer_TCG':2,'tc_summer_TCW':3,
        'tc_fall_TCB':1,'tc_fall_TCG':2,'tc_fall_TCW':3,
    }

    # Multi-band rasters — open once, extract all bands needed
    MULTIBAND = {
        's2_spring': f'{COV_DIR}/s2_spring_12bands.tif',
        's2_summer': f'{COV_DIR}/s2_summer_12bands.tif',
        's2_fall':   f'{COV_DIR}/s2_fall_12bands.tif',
        'tc_spring': f'{COV_DIR}/tc_spring_merged_5070.tif',
        'tc_summer': f'{COV_DIR}/tc_summer_merged_5070.tif',
        'tc_fall':   f'{COV_DIR}/tc_fall_merged_5070.tif',
    }

    # Single-band rasters
    SINGLE = {
        'minnesota_dem_10m':               f'{COV_DIR}/minnesota_dem_10m.tif',
        'relativeTopographicPosition_4m':  f'{COV_DIR}/relativeTopographicPosition_4m.tif',
        'relativeTopographicPosition_8m':  f'{COV_DIR}/relativeTopographicPosition_8m.tif',
        'relativeTopographicPosition_16m': f'{COV_DIR}/relativeTopographicPosition_16m.tif',
        'dist_to_water_10m':               f'{COV_DIR}/dist_to_water_10m.tif',
        'dist_to_stream_10m':              f'{COV_DIR}/dist_to_stream_10m.tif',
        'dist_to_road_detailed_10m':       f'{COV_DIR}/dist_to_road_detailed_10m.tif',
        'dist_from_waterbody_edge_10m':    f'{COV_DIR}/dist_from_waterbody_edge_10m.tif',
        'prism_ppt_mn':                    f'{COV_DIR}/prism_ppt_mn.tif',
        'prism_tmax_july_mn':              f'{COV_DIR}/prism_tmax_july_mn.tif',
        'prism_tmean_mn':                  f'{COV_DIR}/prism_tmean_mn.tif',
        'prism_tmin_january_mn':           f'{COV_DIR}/prism_tmin_january_mn.tif',
        'slope':                           f'{COV_DIR}/slope.tif',
        'aspect':                          f'{COV_DIR}/aspect.tif',
        'wetnessIndex':                    f'{COV_DIR}/wetnessIndex.tif',
        'diffFromMeanElev':                f'{COV_DIR}/diffFromMeanElev.tif',
        'devfrommeanelev_4m':              f'{COV_DIR}/devfrommeanelev_4m.tif',
        'devfrommeanelev_8m':              f'{COV_DIR}/devfrommeanelev_8m.tif',
        'devfrommeanelev_16m':             f'{COV_DIR}/devfrommeanelev_16m.tif',
        'planCurvature':                   f'{COV_DIR}/planCurvature.tif',
        'profileCurvature':                f'{COV_DIR}/profileCurvature.tif',
        'maximalCurvature':                f'{COV_DIR}/maximalCurvature.tif',
        'geomorphons':                     f'{COV_DIR}/geomorphons.tif',
        'mn_nwi_cowardin_raw':             f'{COV_DIR}/mn_nwi_cowardin_10m.tif',
        'exp410_prob':                     f'{PRED_DIR}/exp410_statewide/exp410_prob_minnesota_regrid.tif',
    }

    n_done = 0
    n_total = len(SINGLE) + len(MULTIBAND)

    # Single-band — open once per raster, sample all points
    for name, path in SINGLE.items():
        if not os.path.exists(path):
            print(f'  MISSING: {name}')
            feat_df[name] = np.nan
            n_done += 1
            continue
        with rasterio.open(path) as src:
            vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
            nd   = src.nodata
            if nd is not None: vals = np.where(vals==nd, np.nan, vals)
        if name == 'dist_from_waterbody_edge_10m':
            vals = np.where(np.isnan(vals), 99999, vals)
        if name == 'mn_nwi_cowardin_raw':
            # Convert to binary wetland
            feat_df['mn_nwi_binary'] = ((vals==1)|(vals==2)).astype(float)
            n_done += 1
            elapsed = time.time() - t0
            print(f'  [{n_done}/{n_total}] mn_nwi_binary  ({elapsed:.0f}s elapsed)')
            continue
        feat_df[name] = vals
        n_done += 1
        elapsed = time.time() - t0
        print(f'  [{n_done}/{n_total}] {name}  nan={np.isnan(vals).sum():,}  ({elapsed:.0f}s)')

    # Multi-band — open each raster once, extract all needed bands
    for prefix, path in MULTIBAND.items():
        if not os.path.exists(path):
            print(f'  MISSING: {prefix}')
            n_done += 1
            continue
        # Find all bands needed from this raster
        needed = {feat: band for feat, band in BAND_MAP.items()
                  if feat.startswith(prefix)}
        with rasterio.open(path) as src:
            # Sample all points — returns array of shape (n_points, n_bands)
            samples = np.array(list(src.sample(coords)), dtype=np.float32)
            nd      = src.nodata
            if nd is not None: samples = np.where(samples==nd, np.nan, samples)
        for feat_name, band_idx in needed.items():
            feat_df[feat_name] = samples[:, band_idx-1]
        n_done += 1
        elapsed = time.time() - t0
        print(f'  [{n_done}/{n_total}] {prefix} ({len(needed)} bands)  ({elapsed:.0f}s)')

    elapsed = time.time() - t0
    print(f'\nExtraction complete in {elapsed/60:.1f} minutes')
    feat_df.to_csv(EXTRACTED_PATH, index=False)
    print(f'Saved: {EXTRACTED_PATH}')

# ── STEP 3: Prepare ────────────────────────────────────────────────
print('\n' + '='*60)
print('STEP 3: Prepare training data')
print('='*60)

NON_FEAT = ['x','y','C_AGC_KGM2','STAND_KEY','MN_SPP',
             'BASAL_AREA','JENKINS_GR','AREA_ACRES','exp410_prob',
             'mn_nwi_cowardin_raw']
LEAKERS  = ['histosol','npc_peat','MN_organic','MN_ANY','gNATSGO']

base_features = [
    c for c in feat_df.columns
    if c not in NON_FEAT
    and not any(c.startswith(l) for l in LEAKERS)
    and feat_df[c].dtype in [np.float64, np.float32,
                               np.int64, np.int32, float, int]
]
nan_pcts  = feat_df[base_features].isna().mean()
bad_feats = nan_pcts[nan_pcts > 0.30].index.tolist()
if bad_feats:
    print(f'Dropping {len(bad_feats)} features >30% NaN: {bad_feats}')
    base_features = [f for f in base_features if f not in bad_feats]

df_train = feat_df.dropna(subset=base_features + [TARGET]).copy()
print(f'Training rows: {len(df_train):,}')
print(f'Unique polygons: {df_train["STAND_KEY"].nunique():,}')
print(f'Base features: {len(base_features)}')
print(f'Target mean: {df_train[TARGET].mean():.3f} kgC/m2  '
      f'std={df_train[TARGET].std():.3f}')

X      = df_train[base_features].values
y      = df_train[TARGET].values
groups = df_train['STAND_KEY'].values
gkf    = GroupKFold(n_splits=N_FOLDS)

# ── STEP 4: RFE ────────────────────────────────────────────────────
if os.path.exists(RFE_PATH):
    print(f'\nSTEP 4: Loading existing RFE from {RFE_PATH}')
    with open(RFE_PATH) as f:
        rfe_out = json.load(f)
    best_features = rfe_out['best_features']
    print(f'  Best: {len(best_features)} features  R2={rfe_out["R2"]:.4f}')
else:
    print('\n' + '='*60)
    print('STEP 4: RFE with GroupKFold')
    print('='*60)
    remaining = base_features.copy()
    rfe_curve = []
    DROP_STEP = 10
    MIN_FEATS = 10

    while len(remaining) > MIN_FEATS:
        X_rfe = df_train[remaining].values
        oof   = np.zeros(len(y))
        imps  = np.zeros(len(remaining))
        for tr_idx, va_idx in gkf.split(X_rfe, y, groups):
            rf = RandomForestRegressor(n_estimators=N_TREES_RFE,
                                        n_jobs=-1, random_state=RANDOM_STATE)
            rf.fit(X_rfe[tr_idx], y[tr_idx])
            oof[va_idx] = rf.predict(X_rfe[va_idx])
            imps += rf.feature_importances_
        r2  = r2_score(y, oof)
        mae = mean_absolute_error(y, oof)
        rfe_curve.append({'n_features': len(remaining), 'R2': r2,
                           'MAE': mae, 'features': remaining.copy()})
        print(f'  n={len(remaining):>3}  R2={r2:.4f}  MAE={mae:.4f}')
        n_drop   = min(DROP_STEP, len(remaining) - MIN_FEATS)
        drop_idx = np.argsort(imps)[:n_drop]
        remaining = [f for i,f in enumerate(remaining) if i not in drop_idx]

    X_rfe = df_train[remaining].values
    oof   = np.zeros(len(y))
    for tr_idx, va_idx in gkf.split(X_rfe, y, groups):
        rf = RandomForestRegressor(n_estimators=N_TREES_RFE,
                                    n_jobs=-1, random_state=RANDOM_STATE)
        rf.fit(X_rfe[tr_idx], y[tr_idx])
        oof[va_idx] = rf.predict(X_rfe[va_idx])
    r2  = r2_score(y, oof)
    mae = mean_absolute_error(y, oof)
    rfe_curve.append({'n_features': len(remaining), 'R2': r2,
                       'MAE': mae, 'features': remaining.copy()})
    print(f'  n={len(remaining):>3}  R2={r2:.4f}  MAE={mae:.4f}  <-- final')

    best = max(rfe_curve, key=lambda x: x['R2'])
    best_features = best['features']
    rfe_out = {k:v for k,v in best.items() if k != 'features'}
    rfe_out['best_features'] = best_features
    rfe_out['rfe_curve'] = [{k:v for k,v in r.items() if k!='features'}
                              for r in rfe_curve]
    with open(RFE_PATH,'w') as f:
        json.dump(rfe_out, f, indent=2)
    print(f'  Best: {len(best_features)} features  R2={best["R2"]:.4f}')
    print(f'  Saved: {RFE_PATH}')

    ns  = [r['n_features'] for r in rfe_curve]
    r2s = [r['R2'] for r in rfe_curve]
    plt.figure(figsize=(10,4))
    plt.plot(ns, r2s, 'o-', color='green')
    plt.axvline(len(best_features), color='red', linestyle='--')
    plt.xlabel('Features'); plt.ylabel('GroupKFold R²')
    plt.title('AGC RFE Curve'); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{RES_DIR}/rfe_agc_curve.png', dpi=150)
    plt.close()

# ── STEP 5: Final model ────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    print(f'\nSTEP 5: Model already exists at {MODEL_PATH} — skipping')
else:
    print('\n' + '='*60)
    print('STEP 5: Train final model')
    print('='*60)
    X_best    = df_train[best_features].values
    oof_final = np.zeros(len(y))
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_best, y, groups)):
        rf = RandomForestRegressor(n_estimators=N_TREES_FINAL,
                                    n_jobs=-1, random_state=RANDOM_STATE)
        rf.fit(X_best[tr_idx], y[tr_idx])
        oof_final[va_idx] = rf.predict(X_best[va_idx])
        print(f'  Fold {fold_i+1}/5 done')

    r2_final  = r2_score(y, oof_final)
    mae_final = mean_absolute_error(y, oof_final)
    print(f'  CV R2={r2_final:.4f}  MAE={mae_final:.4f} kgC/m2')

    rf_final = RandomForestRegressor(n_estimators=N_TREES_FINAL,
                                      n_jobs=-1, random_state=RANDOM_STATE)
    rf_final.fit(X_best, y)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(rf_final, f)
    with open(FEAT_PATH, 'w') as f:
        json.dump({'features': best_features, 'target': TARGET}, f, indent=2)
    print(f'  Saved model: {MODEL_PATH}')

    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    axes[0].scatter(y, oof_final, alpha=0.2, s=3, color='green')
    lim = [0, max(y.max(), oof_final.max())*1.05]
    axes[0].plot(lim, lim, 'r--', lw=1.5)
    axes[0].set_xlabel('Observed (kgC/m²)')
    axes[0].set_ylabel('Predicted (kgC/m²)')
    axes[0].set_title(f'AGC RF  R²={r2_final:.4f}  MAE={mae_final:.4f} kgC/m²')
    axes[0].grid(alpha=0.3)
    fi = pd.DataFrame({'feature': best_features,
                        'importance': rf_final.feature_importances_})\
           .sort_values('importance', ascending=False).head(15)
    axes[1].barh(range(len(fi)), fi['importance'].values[::-1],
                  color='green', alpha=0.8)
    axes[1].set_yticks(range(len(fi)))
    axes[1].set_yticklabels([f[:35] for f in fi['feature'].values[::-1]], fontsize=8)
    axes[1].set_xlabel('Importance')
    axes[1].set_title('Top 15 Features')
    axes[1].grid(alpha=0.3, axis='x')
    plt.suptitle('Aboveground Carbon RF Model', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{RES_DIR}/agc_rf_results.png', dpi=150)
    plt.close()

print('\n' + '='*60)
print('ALL STEPS COMPLETE')
print(f'  Model:    {MODEL_PATH}')
print(f'  Features: {FEAT_PATH}')
print(f'  Next:     sbatch run_agc_rf_inference.sh')
print('='*60)
ENDPYTHON

echo ""
echo "=============================="
echo "Finished: $(date)"
echo "=============================="
