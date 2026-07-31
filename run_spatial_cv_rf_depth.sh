#!/bin/bash
#SBATCH --job-name=spatial_rf_depth
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/spatial_rf_depth_%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/spatial_rf_depth_%j.err

echo "=============================="
echo "Spatial CV — RF Peat Depth"
echo "50km spatial blocks, 5 folds"
echo "Started: $(date)"
echo "=============================="

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os, json, pickle
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

BASE_M   = '/scratch.global/ocon0444/peat_modeling'
PROC_DIR = f'{BASE_M}/00_data/processed'
MDL_DIR  = f'{BASE_M}/03_models/exp416'
RES_DIR  = f'{BASE_M}/05_results/exp416_spatial_cv'
EXP410   = f'{BASE_M}/04_predictions/exp410_statewide/exp410_prob_minnesota_regrid.tif'
os.makedirs(RES_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS      = 5
BLOCK_SIZE   = 50000
PROB_THRESH  = 0.33

with open(f'{MDL_DIR}/feature_list.json') as f:
    features = json.load(f)['features']
print('Features: ' + str(len(features)))

df = pd.read_csv(f'{PROC_DIR}/depb_features_extracted.csv', low_memory=False)
df = df[df['depb'].notna() & (df['depb'] > 0)].reset_index(drop=True)

# Filter to exp410 >= 0.33
t_proj = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
xs, ys = t_proj.transform(df['long'].values, df['lat'].values)
with rasterio.open(EXP410) as src:
    tf = src.transform
    probs = []
    for x, y_coord in zip(xs, ys):
        try:
            r, c = rowcol(tf, x, y_coord)
            val = float(src.read(1, window=rasterio.windows.Window(c,r,1,1))[0][0])
        except:
            val = 0.0
        probs.append(val)
df['exp410_prob'] = probs
df = df[df['exp410_prob'] >= PROB_THRESH].reset_index(drop=True)
xs = xs[df.index] if len(xs) != len(df) else \
     np.array([t_proj.transform(lon, lat)[0] for lon, lat in zip(df['long'].values, df['lat'].values)])
ys_coord = np.array([t_proj.transform(lon, lat)[1] for lon, lat in zip(df['long'].values, df['lat'].values)])

features = [f for f in features if f in df.columns]
for feat in features:
    if df[feat].isna().any():
        df[feat] = df[feat].fillna(df[feat].median())
df = df.dropna(subset=features + ['depb']).reset_index(drop=True)
X = df[features].values.astype(np.float32)
y = df['depb'].values.astype(np.float32)
print('n=' + str(len(y)) + '  mean=' + str(round(y.mean(), 1)))

# Reproject for block assignment
t2 = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
xs2, ys2 = t2.transform(df['long'].values, df['lat'].values)
block_x   = (xs2 // BLOCK_SIZE).astype(int)
block_y   = (ys2 // BLOCK_SIZE).astype(int)
block_ids = block_x * 10000 + block_y
unique_blocks = np.unique(block_ids)
np.random.RandomState(RANDOM_STATE).shuffle(unique_blocks)
block_to_fold = {b: i % N_FOLDS for i, b in enumerate(unique_blocks)}
fold_ids = np.array([block_to_fold[b] for b in block_ids])
print('Unique blocks: ' + str(len(unique_blocks)))

with open(f'{MDL_DIR}/model_fold_0.pkl', 'rb') as f:
    ref_model = pickle.load(f)

oof_pred  = np.zeros(len(y))
fold_r2s  = []

for fold_i in range(N_FOLDS):
    tr_idx = np.where(fold_ids != fold_i)[0]
    va_idx = np.where(fold_ids == fold_i)[0]
    print('Fold ' + str(fold_i+1) + '/5  tr=' + str(len(tr_idx)) + '  va=' + str(len(va_idx)))

    model = RandomForestRegressor(
        n_estimators=ref_model.n_estimators,
        max_features=ref_model.max_features,
        min_samples_leaf=ref_model.min_samples_leaf,
        n_jobs=16, random_state=RANDOM_STATE
    )
    model.fit(X[tr_idx], y[tr_idx])
    oof_pred[va_idx] = model.predict(X[va_idx])
    fold_r2 = r2_score(y[va_idx], oof_pred[va_idx])
    fold_r2s.append(fold_r2)
    print('  R2=' + str(round(fold_r2, 4)))

r2  = r2_score(y, oof_pred)
mae = mean_absolute_error(y, oof_pred)

print('\nRF Depth Spatial CV:')
print('  R2:    ' + str(round(r2, 4)))
print('  MAE:   ' + str(round(mae, 1)) + ' cm')
print('  Folds: ' + str([round(r, 4) for r in fold_r2s]))
print('  vs random CV R2: 0.484')

np.save(f'{RES_DIR}/oof_pred.npy', oof_pred)
np.save(f'{RES_DIR}/oof_true.npy', y)
with open(f'{RES_DIR}/spatial_cv_results.json', 'w') as f:
    json.dump({'model':'RF','target':'depth','cv':'spatial_block_50km',
               'n_folds':N_FOLDS,'r2':r2,'mae':mae,
               'fold_r2s':fold_r2s,'n':len(y)}, f, indent=2)
print('Saved to ' + RES_DIR)
ENDPYTHON

echo "Finished: $(date)"
