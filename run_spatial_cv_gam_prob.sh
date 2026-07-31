#!/bin/bash
#SBATCH --job-name=spatial_gam_prob
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/spatial_gam_prob_%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/spatial_gam_prob_%j.err

echo "=============================="
echo "Spatial CV — GAM Peat Probability"
echo "50km spatial blocks, 5 folds"
echo "pygam LogisticGAM, 30k subsample per fold"
echo "Started: $(date)"
echo "=============================="

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os, json, pickle
import numpy as np
import pandas as pd
from pygam import LogisticGAM, s, f
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

BASE_M   = '/scratch.global/ocon0444/peat_modeling'
PROC_DIR = f'{BASE_M}/00_data/processed'
MDL_DIR  = f'{BASE_M}/03_models/exp410'
RES_DIR  = f'{BASE_M}/05_results/exp_gam_peat_spatial_cv'
MDL_OUT  = f'{BASE_M}/03_models/exp_gam_peat_spatial_cv'
for d in [RES_DIR, MDL_OUT]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS      = 5
BLOCK_SIZE   = 50000
PROB_THRESH  = 0.33
N_SUBSAMPLE  = 30000  # GAM is slow — subsample per fold

with open(f'{MDL_DIR}/feature_list.json') as f:
    features = json.load(f)['features']
if 'mn_nwi_binary' not in [f for f in features]:
    pass

df = pd.read_csv(f'{PROC_DIR}/binary_peat_features_0_20_dropped.csv', low_memory=False)
if 'mn_nwi_binary' in features and 'mn_nwi_binary' not in df.columns:
    df['mn_nwi_binary'] = df['mn_nwi_merged_1_2']
features = [f for f in features if f in df.columns]
df = df.dropna(subset=['peat_binary'] + features).reset_index(drop=True)
X_raw = df[features].values.astype(np.float32)
y     = df['peat_binary'].values.astype(int)
print('n=' + str(len(y)) + '  peat=' + str(y.sum()) + '  features=' + str(len(features)))

# Scale — required for GAM
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Spatial block folds
t = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
xs, ys = t.transform(df['long'].values, df['lat'].values)
block_x   = (xs // BLOCK_SIZE).astype(int)
block_y   = (ys // BLOCK_SIZE).astype(int)
block_ids = block_x * 10000 + block_y
unique_blocks = np.unique(block_ids)
np.random.RandomState(RANDOM_STATE).shuffle(unique_blocks)
block_to_fold = {b: i % N_FOLDS for i, b in enumerate(unique_blocks)}
fold_ids = np.array([block_to_fold[b] for b in block_ids])
print('Blocks: ' + str(len(unique_blocks)))
for i in range(N_FOLDS):
    print('  Fold ' + str(i) + ': ' + str((fold_ids==i).sum()) + ' points')

# Build GAM terms — spline for each continuous feature
n_feat = len(features)
terms  = s(0)
for i in range(1, n_feat):
    terms += s(i)

oof_prob  = np.zeros(len(y))
fold_aucs = []

for fold_i in range(N_FOLDS):
    tr_idx = np.where(fold_ids != fold_i)[0]
    va_idx = np.where(fold_ids == fold_i)[0]

    # Balanced subsample of training fold
    tr_peat    = tr_idx[y[tr_idx]==1]
    tr_nonpeat = tr_idx[y[tr_idx]==0]
    n_each     = min(N_SUBSAMPLE//2, len(tr_peat), len(tr_nonpeat))
    rng        = np.random.RandomState(RANDOM_STATE + fold_i)
    sub_tr     = np.concatenate([
        rng.choice(tr_peat,    n_each, replace=False),
        rng.choice(tr_nonpeat, n_each, replace=False)
    ])
    print('Fold ' + str(fold_i+1) + '/5  train_sub=' + str(len(sub_tr)) +
          '  va=' + str(len(va_idx)))

    try:
        gam = LogisticGAM(terms, max_iter=100)
        gam.gridsearch(X[sub_tr], y[sub_tr],
                       lam=np.logspace(-3, 3, 10),
                       progress=False)
        oof_prob[va_idx] = gam.predict_proba(X[va_idx])
    except Exception as e:
        print('  Gridsearch failed (' + str(e) + '), using fixed lam=0.1')
        try:
            gam = LogisticGAM(terms, lam=0.1, max_iter=200)
            gam.fit(X[sub_tr], y[sub_tr])
            oof_prob[va_idx] = gam.predict_proba(X[va_idx])
        except Exception as e2:
            print('  Fixed lam also failed: ' + str(e2))
            oof_prob[va_idx] = np.mean(y[sub_tr])
            fold_aucs.append(0.5)
            continue

    fold_auc = roc_auc_score(y[va_idx], oof_prob[va_idx])
    fold_aucs.append(fold_auc)
    print('  AUC=' + str(round(fold_auc, 4)))

auc = roc_auc_score(y, oof_prob)
ap  = average_precision_score(y, oof_prob)
f1  = f1_score(y, (oof_prob >= PROB_THRESH).astype(int))

print('\nGAM Probability Spatial CV:')
print('  AUC:   ' + str(round(auc, 4)))
print('  AvgPr: ' + str(round(ap, 4)))
print('  F1:    ' + str(round(f1, 4)))
print('  Folds: ' + str([round(a, 4) for a in fold_aucs]))
print('  vs random CV AUC: 0.9915')

# Train final model on balanced subsample of full dataset
rng_f   = np.random.RandomState(RANDOM_STATE)
all_p   = np.where(y==1)[0]
all_np  = np.where(y==0)[0]
n_fin   = min(N_SUBSAMPLE//2, len(all_p), len(all_np))
fin_sub = np.concatenate([rng_f.choice(all_p, n_fin, replace=False),
                           rng_f.choice(all_np, n_fin, replace=False)])
try:
    gam_final = LogisticGAM(terms, max_iter=100)
    gam_final.gridsearch(X[fin_sub], y[fin_sub],
                         lam=np.logspace(-3, 3, 10), progress=False)
except:
    gam_final = LogisticGAM(terms, lam=0.1, max_iter=200)
    gam_final.fit(X[fin_sub], y[fin_sub])

pickle.dump({'model':gam_final,'scaler':scaler,'features':features},
            open(f'{MDL_OUT}/gam_prob_final.pkl','wb'))
np.save(f'{RES_DIR}/gam_oof_prob.npy', oof_prob)
np.save(f'{RES_DIR}/gam_oof_true.npy', y)
with open(f'{RES_DIR}/spatial_cv_results.json','w') as f:
    json.dump({'model':'GAM_logistic','target':'probability',
               'cv':'spatial_block_50km','n_folds':N_FOLDS,
               'auc':auc,'avg_precision':ap,'f1':f1,
               'fold_aucs':fold_aucs,'n':len(y),
               'train_sub':int(N_SUBSAMPLE)}, f, indent=2)
print('Saved to ' + RES_DIR)
print('DONE')
ENDPYTHON

echo "Finished: $(date)"
