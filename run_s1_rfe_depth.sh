#!/bin/bash
#SBATCH --job-name=s1_rfe_depth
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=6:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/s1_rfe_depth_%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/s1_rfe_depth_%j.err

echo "S1 RFE — Depth (RF, random + spatial CV)"
echo "Started: $(date)"

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os, json, pickle, time
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

BASE_M        = '/scratch.global/ocon0444/peat_modeling'
CSV_PATH      = f'{BASE_M}/00_data/processed/depb_features_extracted_s1.csv'
ORIG_FEAT_JSON= f'{BASE_M}/03_models/depth/DEPTH_RF_001/feature_list.json'
EXP410        = f'{BASE_M}/04_predictions/exp410_statewide/exp410_prob_minnesota_regrid.tif'
OUT_DIR       = f'{BASE_M}/05_results/depth/rfe/s1_rfe'
MDL_DIR       = f'{BASE_M}/03_models/depth/DEPTH_RF_S1_RFE'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)

RANDOM_STATE  = 42
N_FOLDS       = 5
N_TREES       = 200
R2_TOLERANCE  = 0.01
BLOCK_SIZE    = 50000
PROB_THRESH   = 0.5

# Descending passes empty over MN — ascending only (9 features)
S1_FEATURES = []
for comp in ['s1_spring_asc','s1_summer_asc','s1_fall_asc']:
    for band in ['VV','VH','VV_VH_ratio']:
        S1_FEATURES.append(f'{comp}_{band}')

with open(ORIG_FEAT_JSON) as f:
    orig_features = json.load(f)['features']
start_features = orig_features + S1_FEATURES
print('Starting features: ' + str(len(start_features)))

# Load and filter depth data
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df[df['depb'].notna() & (df['depb'] > 0)].reset_index(drop=True)

# Apply exp410 >= 0.5 mask
t = Transformer.from_crs('EPSG:4326','EPSG:5070',always_xy=True)
xs, ys = t.transform(df['long'].values, df['lat'].values)
with rasterio.open(EXP410) as src:
    probs = np.array([v[0] for v in src.sample(zip(xs,ys), indexes=1)])
probs = np.where(np.isnan(probs), 0, probs)
df['exp410_prob'] = probs
df = df[df['exp410_prob'] >= PROB_THRESH].reset_index(drop=True)
print('After exp410>=0.5 filter: n=' + str(len(df)))

y = df['depb'].values.astype(np.float32)

# Spatial blocks
xs2, ys2  = t.transform(df['long'].values, df['lat'].values)
block_x   = (xs2 // BLOCK_SIZE).astype(int)
block_y   = (ys2 // BLOCK_SIZE).astype(int)
block_ids = block_x * 10000 + block_y
unique_blocks = np.unique(block_ids)
np.random.RandomState(RANDOM_STATE).shuffle(unique_blocks)
block_to_fold = {b: i % N_FOLDS for i,b in enumerate(unique_blocks)}
spatial_fold_ids = np.array([block_to_fold[b] for b in block_ids])

def prepare_X(df, cols):
    X = df[cols].apply(pd.to_numeric, errors='coerce')
    return X.fillna(X.median())

def run_cv_random(df, y, features):
    X  = prepare_X(df, features)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r2s = []; imps = np.zeros(len(features))
    for tr_idx, va_idx in kf.split(X):
        rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=16,
                                   random_state=RANDOM_STATE)
        rf.fit(X.iloc[tr_idx], y[tr_idx])
        pred = rf.predict(X.iloc[va_idx])
        r2s.append(r2_score(y[va_idx], pred))
        imps += rf.feature_importances_
    return float(np.mean(r2s)), imps / N_FOLDS

def run_cv_spatial(df, y, features, fold_ids):
    X = prepare_X(df, features).values
    r2s = []; imps = np.zeros(len(features))
    for fold_i in range(N_FOLDS):
        tr_idx = np.where(fold_ids != fold_i)[0]
        va_idx = np.where(fold_ids == fold_i)[0]
        rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=16,
                                   random_state=RANDOM_STATE)
        rf.fit(X[tr_idx], y[tr_idx])
        pred = rf.predict(X[va_idx])
        r2s.append(r2_score(y[va_idx], pred))
        imps += rf.feature_importances_
    return float(np.mean(r2s)), imps / N_FOLDS

# RFE Loop
print('\nStarting depth RFE...')
features     = [f for f in start_features if f in df.columns]
history      = []
best_rand_r2 = -999
best_spat_r2 = -999
best_features= list(features)
t0 = time.time()

while len(features) > 5:
    rand_r2, rand_imp = run_cv_random(df, y, features)
    spat_r2, spat_imp = run_cv_spatial(df, y, features, spatial_fold_ids)

    entry = {
        'n_features': len(features),
        'rand_r2':    round(rand_r2, 4),
        'spat_r2':    round(spat_r2, 4),
        'features':   list(features),
        'dropped':    None,
    }

    if rand_r2 > best_rand_r2:
        best_rand_r2 = rand_r2
        best_features = list(features)
    if spat_r2 > best_spat_r2:
        best_spat_r2 = spat_r2

    elapsed = round(time.time()-t0, 1)
    print(f'  n={len(features):3d}  rand_R2={rand_r2:.4f}  spat_R2={spat_r2:.4f}  ({elapsed}s)')

    if best_rand_r2 - rand_r2 > R2_TOLERANCE:
        print('  Stopping: R2 drop exceeded tolerance')
        break

    drop_idx  = int(np.argmin(rand_imp))
    drop_feat = features[drop_idx]
    entry['dropped'] = drop_feat
    history.append(entry)
    features.pop(drop_idx)

print('\nDepth RFE complete.')
print('Best random CV R2: ' + str(round(best_rand_r2, 4)))
print('Best spatial CV R2: ' + str(round(best_spat_r2, 4)))
print('Best n features: ' + str(len(best_features)))

s1_survived = [f for f in best_features if f.startswith('s1_')]
print('S1 features survived: ' + str(len(s1_survived)))
for f in s1_survived:
    print('  ' + f)

# Train final model
X_best = prepare_X(df, best_features).values
rf_final = RandomForestRegressor(n_estimators=500, n_jobs=16,
                                  random_state=RANDOM_STATE)
rf_final.fit(X_best, y)
pickle.dump(rf_final, open(f'{MDL_DIR}/rf_depth_s1_final.pkl','wb'))

with open(f'{MDL_DIR}/feature_list.json','w') as f:
    json.dump({
        'features':    best_features,
        'model_code':  'DEPTH_RF_S1_001',
        'n_features':  len(best_features),
        's1_survived': s1_survived,
        'rand_r2':     best_rand_r2,
        'spat_r2':     best_spat_r2,
        'starting_n':  len(start_features),
    }, f, indent=2)

pd.DataFrame(history).to_csv(f'{OUT_DIR}/rfe_history.csv', index=False)
print('Saved to ' + MDL_DIR)
print('DONE')
ENDPYTHON

echo "Finished: $(date)"
