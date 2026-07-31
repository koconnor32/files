#!/bin/bash
#SBATCH --job-name=spatial_lgbm_depth
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/spatial_lgbm_depth_%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/spatial_lgbm_depth_%j.err

echo "=============================="
echo "Spatial CV — LightGBM Peat Depth"
echo "50km spatial blocks, 5 folds"
echo "Started: $(date)"
echo "=============================="

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os, json, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

BASE_M   = '/scratch.global/ocon0444/peat_modeling'
PROC_DIR = f'{BASE_M}/00_data/processed'
MDL_DIR  = f'{BASE_M}/03_models/exp_lgbm_depth'
RES_DIR  = f'{BASE_M}/05_results/exp_lgbm_depth_spatial_cv'
EXP410   = f'{BASE_M}/04_predictions/exp410_statewide/exp410_prob_minnesota_regrid.tif'
os.makedirs(RES_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS      = 5
BLOCK_SIZE   = 50000
PROB_THRESH  = 0.33

with open(f'{MDL_DIR}/lgbm_depth_features.json') as f:
    features = json.load(f)['features']
print('Features: ' + str(len(features)))

df = pd.read_csv(f'{PROC_DIR}/depb_features_extracted.csv', low_memory=False)
df = df[df['depb'].notna() & (df['depb'] > 0)].reset_index(drop=True)

t = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
xs, ys = t.transform(df['long'].values, df['lat'].values)
with rasterio.open(EXP410) as src:
    tf = src.transform
    probs = []
    for x, y_c in zip(xs, ys):
        try:
            r, c = rowcol(tf, x, y_c)
            val = float(src.read(1, window=rasterio.windows.Window(c,r,1,1))[0][0])
        except:
            val = 0.0
        probs.append(val)
df['exp410_prob'] = probs
df = df[df['exp410_prob'] >= PROB_THRESH].reset_index(drop=True)

features = [f for f in features if f in df.columns]
for feat in features:
    if df[feat].isna().any():
        df[feat] = df[feat].fillna(df[feat].median())
df = df.dropna(subset=features + ['depb']).reset_index(drop=True)
X = df[features].values.astype(np.float32)
y = df['depb'].values.astype(np.float32)
print('n=' + str(len(y)) + '  mean=' + str(round(y.mean(), 1)))

xs2, ys2 = t.transform(df['long'].values, df['lat'].values)
block_x   = (xs2 // BLOCK_SIZE).astype(int)
block_y   = (ys2 // BLOCK_SIZE).astype(int)
block_ids = block_x * 10000 + block_y
unique_blocks = np.unique(block_ids)
np.random.RandomState(RANDOM_STATE).shuffle(unique_blocks)
block_to_fold = {b: i % N_FOLDS for i, b in enumerate(unique_blocks)}
fold_ids = np.array([block_to_fold[b] for b in block_ids])
print('Unique blocks: ' + str(len(unique_blocks)))

with open(f'{MDL_DIR}/lgbm_depth_final.pkl', 'rb') as f:
    ref_model = pickle.load(f)
params = ref_model.get_params()

oof_pred = np.zeros(len(y))
fold_r2s = []

for fold_i in range(N_FOLDS):
    tr_idx = np.where(fold_ids != fold_i)[0]
    va_idx = np.where(fold_ids == fold_i)[0]
    print('Fold ' + str(fold_i+1) + '/5  tr=' + str(len(tr_idx)) + '  va=' + str(len(va_idx)))

    model = lgb.LGBMRegressor(
        n_estimators=params.get('n_estimators', 500),
        max_depth=params.get('max_depth', -1),
        learning_rate=params.get('learning_rate', 0.05),
        num_leaves=params.get('num_leaves', 63),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        n_jobs=16, random_state=RANDOM_STATE,
        verbose=-1
    )
    model.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    oof_pred[va_idx] = np.clip(model.predict(X[va_idx]), 0, None)
    fold_r2 = r2_score(y[va_idx], oof_pred[va_idx])
    fold_r2s.append(fold_r2)
    print('  R2=' + str(round(fold_r2, 4)))

r2  = r2_score(y, oof_pred)
mae = mean_absolute_error(y, oof_pred)

print('\nLightGBM Depth Spatial CV:')
print('  R2:    ' + str(round(r2, 4)))
print('  MAE:   ' + str(round(mae, 1)) + ' cm')
print('  Folds: ' + str([round(r, 4) for r in fold_r2s]))
print('  vs random CV R2: ~0.51')

np.save(f'{RES_DIR}/lgbm_oof_pred.npy', oof_pred)
np.save(f'{RES_DIR}/lgbm_oof_true.npy', y)
with open(f'{RES_DIR}/spatial_cv_results.json', 'w') as f:
    json.dump({'model':'LightGBM','target':'depth','cv':'spatial_block_50km',
               'n_folds':N_FOLDS,'r2':r2,'mae':mae,
               'fold_r2s':fold_r2s,'n':len(y)}, f, indent=2)
print('Saved to ' + RES_DIR)
ENDPYTHON

echo "Finished: $(date)"
