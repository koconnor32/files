#!/bin/bash
#SBATCH --job-name=spatial_rf_prob
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=4:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/spatial_rf_prob_%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/spatial_rf_prob_%j.err

echo "=============================="
echo "Spatial CV — RF Peat Probability"
echo "50km spatial blocks, 5 folds"
echo "Started: $(date)"
echo "=============================="

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

BASE_M   = '/scratch.global/ocon0444/peat_modeling'
PROC_DIR = f'{BASE_M}/00_data/processed'
MDL_DIR  = f'{BASE_M}/03_models/exp410'
RES_DIR  = f'{BASE_M}/05_results/exp410_spatial_cv'
os.makedirs(RES_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS      = 5
BLOCK_SIZE   = 50000
PROB_THRESH  = 0.33

# Load features from exp410
with open(f'{MDL_DIR}/feature_list.json') as f:
    features = json.load(f)['features']
print('Features: ' + str(len(features)))

df = pd.read_csv(f'{PROC_DIR}/binary_peat_features_0_20_dropped.csv', low_memory=False)
if 'mn_nwi_binary' in features and 'mn_nwi_binary' not in df.columns:
    df['mn_nwi_binary'] = df['mn_nwi_merged_1_2']
features = [f for f in features if f in df.columns]
df = df.dropna(subset=['peat_binary'] + features).reset_index(drop=True)
X = df[features].values.astype(np.float32)
y = df['peat_binary'].values.astype(int)
print('n=' + str(len(y)) + '  peat=' + str(y.sum()))

# Spatial block fold assignment
t = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
xs, ys = t.transform(df['long'].values, df['lat'].values)
block_x   = (xs // BLOCK_SIZE).astype(int)
block_y   = (ys // BLOCK_SIZE).astype(int)
block_ids = block_x * 10000 + block_y
unique_blocks = np.unique(block_ids)
np.random.RandomState(RANDOM_STATE).shuffle(unique_blocks)
block_to_fold = {b: i % N_FOLDS for i, b in enumerate(unique_blocks)}
fold_ids = np.array([block_to_fold[b] for b in block_ids])
print('Unique blocks: ' + str(len(unique_blocks)))
for i in range(N_FOLDS):
    print('  Fold ' + str(i) + ': ' + str((fold_ids==i).sum()) + ' points')

# Load RF hyperparams from existing model
import pickle
with open(f'{MDL_DIR}/model_fold_0.pkl', 'rb') as f:
    ref_model = pickle.load(f)

oof_prob  = np.zeros(len(y))
fold_aucs = []

for fold_i in range(N_FOLDS):
    tr_idx = np.where(fold_ids != fold_i)[0]
    va_idx = np.where(fold_ids == fold_i)[0]
    print('Fold ' + str(fold_i+1) + '/5  tr=' + str(len(tr_idx)) + '  va=' + str(len(va_idx)))

    model = RandomForestClassifier(
        n_estimators=ref_model.n_estimators,
        max_features=ref_model.max_features,
        min_samples_leaf=ref_model.min_samples_leaf,
        n_jobs=16, random_state=RANDOM_STATE
    )
    model.fit(X[tr_idx], y[tr_idx])
    oof_prob[va_idx] = model.predict_proba(X[va_idx])[:, 1]
    fold_auc = roc_auc_score(y[va_idx], oof_prob[va_idx])
    fold_aucs.append(fold_auc)
    print('  AUC=' + str(round(fold_auc, 4)))

auc = roc_auc_score(y, oof_prob)
ap  = average_precision_score(y, oof_prob)
f1  = f1_score(y, (oof_prob >= PROB_THRESH).astype(int))

print('\nRF Probability Spatial CV:')
print('  AUC:   ' + str(round(auc, 4)))
print('  AvgPr: ' + str(round(ap, 4)))
print('  F1:    ' + str(round(f1, 4)))
print('  Folds: ' + str([round(a, 4) for a in fold_aucs]))
print('  vs random CV AUC: 0.9722')

np.save(f'{RES_DIR}/oof_prob.npy', oof_prob)
np.save(f'{RES_DIR}/oof_true.npy', y)
with open(f'{RES_DIR}/spatial_cv_results.json', 'w') as f:
    json.dump({'model':'RF','target':'probability','cv':'spatial_block_50km',
               'n_folds':N_FOLDS,'auc':auc,'avg_precision':ap,'f1':f1,
               'fold_aucs':fold_aucs,'n':len(y)}, f, indent=2)
print('Saved to ' + RES_DIR)
ENDPYTHON

echo "Finished: $(date)"
