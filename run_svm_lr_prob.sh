#!/bin/bash
#SBATCH --job-name=svm_lr_prob
#SBATCH --partition=agsmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --output=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/svm_lr_prob_%j.out
#SBATCH --error=/scratch.global/ocon0444/peat_modeling/06_sbatch_jobs/logs/svm_lr_prob_%j.err

echo "=============================="
echo "SVM (RBF) + Logistic Regression — Peat Probability"
echo "Features: exp410 41-feature set"
echo "Random 5-fold CV"
echo "Started: $(date)"
echo "=============================="

eval "$(conda shell.bash hook)"
conda activate gdalenvgeospat

python3 - << 'ENDPYTHON'
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

BASE_M   = '/scratch.global/ocon0444/peat_modeling'
PROC_DIR = f'{BASE_M}/00_data/processed'
MDL_DIR  = f'{BASE_M}/03_models/exp410'
RES_SVM  = f'{BASE_M}/05_results/exp_svm_prob'
RES_LR   = f'{BASE_M}/05_results/exp_lr_prob'
MDL_SVM  = f'{BASE_M}/03_models/exp_svm_prob'
MDL_LR   = f'{BASE_M}/03_models/exp_lr_prob'
for d in [RES_SVM, RES_LR, MDL_SVM, MDL_LR]:
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS      = 5
PROB_THRESH  = 0.33
BLOCK_SIZE   = 50000

# Use exp410 features
with open(f'{MDL_DIR}/feature_list.json') as f:
    features = json.load(f)['features']

df = pd.read_csv(f'{PROC_DIR}/binary_peat_features_0_20_dropped.csv', low_memory=False)
if 'mn_nwi_binary' in features and 'mn_nwi_binary' not in df.columns:
    df['mn_nwi_binary'] = df['mn_nwi_merged_1_2']
features = [f for f in features if f in df.columns]
df = df.dropna(subset=['peat_binary'] + features).reset_index(drop=True)
X_raw = df[features].values.astype(np.float32)
y     = df['peat_binary'].values.astype(int)
print('n=' + str(len(y)) + '  peat=' + str(y.sum()) + '  features=' + str(len(features)))

# Scale — required for SVM and LR
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

# ── SVM RBF ───────────────────────────────────────────────────────
print('\n--- SVM RBF ---')
# Subsample for SVM — too slow on full dataset
# Use 15k balanced subsample per fold for training
SVM_N_TRAIN = 15000
svm_oof  = np.zeros(len(y))
svm_aucs = []

for fold_i in range(N_FOLDS):
    tr_idx = np.where(fold_ids != fold_i)[0]
    va_idx = np.where(fold_ids == fold_i)[0]

    # Balanced subsample of training set
    tr_peat    = tr_idx[y[tr_idx]==1]
    tr_nonpeat = tr_idx[y[tr_idx]==0]
    n_each     = min(SVM_N_TRAIN//2, len(tr_peat), len(tr_nonpeat))
    rng        = np.random.RandomState(RANDOM_STATE + fold_i)
    sub_tr     = np.concatenate([
        rng.choice(tr_peat,    n_each, replace=False),
        rng.choice(tr_nonpeat, n_each, replace=False)
    ])
    print('Fold ' + str(fold_i+1) + '/5  train_sub=' + str(len(sub_tr)) + '  va=' + str(len(va_idx)))

    svm = SVC(kernel='rbf', C=1.0, gamma='scale',
              class_weight='balanced', probability=True,
              random_state=RANDOM_STATE)
    svm.fit(X[sub_tr], y[sub_tr])
    svm_oof[va_idx] = svm.predict_proba(X[va_idx])[:, 1]
    fold_auc = roc_auc_score(y[va_idx], svm_oof[va_idx])
    svm_aucs.append(fold_auc)
    print('  AUC=' + str(round(fold_auc, 4)))

svm_auc = roc_auc_score(y, svm_oof)
svm_ap  = average_precision_score(y, svm_oof)
svm_f1  = f1_score(y, (svm_oof >= PROB_THRESH).astype(int))
print('\nSVM Probability Spatial CV:')
print('  AUC:   ' + str(round(svm_auc, 4)))
print('  AvgPr: ' + str(round(svm_ap, 4)))
print('  F1:    ' + str(round(svm_f1, 4)))
print('  Folds: ' + str([round(a, 4) for a in svm_aucs]))

# Train final SVM on full balanced subsample
rng_f = np.random.RandomState(RANDOM_STATE)
all_peat    = np.where(y==1)[0]
all_nonpeat = np.where(y==0)[0]
n_fin = min(SVM_N_TRAIN//2, len(all_peat), len(all_nonpeat))
final_sub = np.concatenate([
    rng_f.choice(all_peat,    n_fin, replace=False),
    rng_f.choice(all_nonpeat, n_fin, replace=False)
])
svm_final = SVC(kernel='rbf', C=1.0, gamma='scale',
                class_weight='balanced', probability=True,
                random_state=RANDOM_STATE)
svm_final.fit(X[final_sub], y[final_sub])
pickle.dump({'model':svm_final,'scaler':scaler,'features':features},
            open(f'{MDL_SVM}/svm_prob_final.pkl','wb'))
np.save(f'{RES_SVM}/svm_oof_prob.npy', svm_oof)
np.save(f'{RES_SVM}/svm_oof_true.npy', y)
with open(f'{RES_SVM}/spatial_cv_results.json','w') as f:
    json.dump({'model':'SVM_RBF','target':'probability','cv':'spatial_block_50km',
               'auc':svm_auc,'avg_precision':svm_ap,'f1':svm_f1,
               'fold_aucs':svm_aucs,'n':len(y),'train_sub':int(SVM_N_TRAIN)}, f, indent=2)
print('SVM saved.')

# ── Logistic Regression ───────────────────────────────────────────
print('\n--- Logistic Regression ---')
lr_oof  = np.zeros(len(y))
lr_aucs = []

for fold_i in range(N_FOLDS):
    tr_idx = np.where(fold_ids != fold_i)[0]
    va_idx = np.where(fold_ids == fold_i)[0]
    print('Fold ' + str(fold_i+1) + '/5  tr=' + str(len(tr_idx)) + '  va=' + str(len(va_idx)))

    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                             solver='lbfgs', n_jobs=16, random_state=RANDOM_STATE)
    lr.fit(X[tr_idx], y[tr_idx])
    lr_oof[va_idx] = lr.predict_proba(X[va_idx])[:, 1]
    fold_auc = roc_auc_score(y[va_idx], lr_oof[va_idx])
    lr_aucs.append(fold_auc)
    print('  AUC=' + str(round(fold_auc, 4)))

lr_auc = roc_auc_score(y, lr_oof)
lr_ap  = average_precision_score(y, lr_oof)
lr_f1  = f1_score(y, (lr_oof >= PROB_THRESH).astype(int))
print('\nLogistic Regression Probability Spatial CV:')
print('  AUC:   ' + str(round(lr_auc, 4)))
print('  AvgPr: ' + str(round(lr_ap, 4)))
print('  F1:    ' + str(round(lr_f1, 4)))
print('  Folds: ' + str([round(a, 4) for a in lr_aucs]))

lr_final = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                               solver='lbfgs', n_jobs=16, random_state=RANDOM_STATE)
lr_final.fit(X, y)
pickle.dump({'model':lr_final,'scaler':scaler,'features':features},
            open(f'{MDL_LR}/lr_prob_final.pkl','wb'))
np.save(f'{RES_LR}/lr_oof_prob.npy', lr_oof)
np.save(f'{RES_LR}/lr_oof_true.npy', y)
with open(f'{RES_LR}/spatial_cv_results.json','w') as f:
    json.dump({'model':'LogisticRegression','target':'probability','cv':'spatial_block_50km',
               'auc':lr_auc,'avg_precision':lr_ap,'f1':lr_f1,
               'fold_aucs':lr_aucs,'n':len(y)}, f, indent=2)
print('LR saved.')

print('\nDONE')
ENDPYTHON

echo "Finished: $(date)"
