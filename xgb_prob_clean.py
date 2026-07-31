import os, json, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

BASE_M   = '/scratch.global/ocon0444/peat_modeling'
PROC_DIR = f'{BASE_M}/00_data/processed'
MDL_OUT  = f'{BASE_M}/03_models/probability/PROB_XGB_002'
RES_OUT  = f'{BASE_M}/05_results/probability/PROB_XGB_002'
os.makedirs(MDL_OUT, exist_ok=True)
os.makedirs(RES_OUT, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS      = 5
BLOCK_SIZE   = 50000
PROB_THRESH  = 0.33

# Use same clean 41-feature set as PROB_RF_001/PROB_CNN_003
with open(f'{BASE_M}/03_models/probability/PROB_RF_001/feature_list.json') as f:
    features = json.load(f)['features']
print('Features (exp410 clean set): ' + str(len(features)))
print('point_id in features:', 'point_id' in features)

df = pd.read_csv(f'{PROC_DIR}/binary_peat_features_0_20_dropped.csv', low_memory=False)
if 'mn_nwi_binary' in features and 'mn_nwi_binary' not in df.columns:
    df['mn_nwi_binary'] = df['mn_nwi_merged_1_2']
features = [f for f in features if f in df.columns]
df = df.dropna(subset=['peat_binary'] + features).reset_index(drop=True)
X = df[features].values.astype(np.float32)
y = df['peat_binary'].values.astype(int)
print('n=' + str(len(y)) + '  peat=' + str(y.sum()) + '  features=' + str(len(features)))

# Get hyperparams from original XGB model
with open(f'{BASE_M}/03_models/probability/PROB_XGB_001/xgb_peat_prob_final.pkl', 'rb') as f:
    ref = pickle.load(f)
params = ref.get_params()
scale_pos = (y==0).sum() / y.sum()

# Spatial block CV
t = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
xs, ys = t.transform(df['long'].values, df['lat'].values)
block_x   = (xs // BLOCK_SIZE).astype(int)
block_y   = (ys // BLOCK_SIZE).astype(int)
block_ids = block_x * 10000 + block_y
unique_blocks = np.unique(block_ids)
np.random.RandomState(RANDOM_STATE).shuffle(unique_blocks)
block_to_fold = {b: i % N_FOLDS for i,b in enumerate(unique_blocks)}
fold_ids = np.array([block_to_fold[b] for b in block_ids])
print('Blocks: ' + str(len(unique_blocks)))

oof_prob  = np.zeros(len(y))
fold_aucs = []

for fold_i in range(N_FOLDS):
    tr_idx = np.where(fold_ids != fold_i)[0]
    va_idx = np.where(fold_ids == fold_i)[0]
    print('Fold ' + str(fold_i+1) + '/5  tr=' + str(len(tr_idx)) + '  va=' + str(len(va_idx)))

    model = xgb.XGBClassifier(
        n_estimators=params.get('n_estimators', 500),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.05),
        subsample=params.get('subsample', 0.8),
        colsample_bytree=params.get('colsample_bytree', 0.8),
        scale_pos_weight=scale_pos,
        n_jobs=16, random_state=RANDOM_STATE,
        eval_metric='auc', verbosity=0
    )
    model.fit(X[tr_idx], y[tr_idx],
              eval_set=[(X[va_idx], y[va_idx])],
              verbose=False)
    oof_prob[va_idx] = model.predict_proba(X[va_idx])[:, 1]
    fold_auc = roc_auc_score(y[va_idx], oof_prob[va_idx])
    fold_aucs.append(fold_auc)
    print('  AUC=' + str(round(fold_auc, 4)))

auc = roc_auc_score(y, oof_prob)
ap  = average_precision_score(y, oof_prob)
f1  = f1_score(y, (oof_prob >= PROB_THRESH).astype(int))

print('\nPROB_XGB_002 (clean 41 features, spatial CV):')
print('  AUC:    ' + str(round(auc, 4)))
print('  AvgPr:  ' + str(round(ap, 4)))
print('  F1:     ' + str(round(f1, 4)))
print('  Folds:  ' + str([round(a, 4) for a in fold_aucs]))
print('  vs PROB_XGB_001 spatial AUC: 0.9630 (had point_id contamination)')

# Train final model on full data
print('Training final model...')
final = xgb.XGBClassifier(
    n_estimators=params.get('n_estimators', 500),
    max_depth=params.get('max_depth', 6),
    learning_rate=params.get('learning_rate', 0.05),
    subsample=params.get('subsample', 0.8),
    colsample_bytree=params.get('colsample_bytree', 0.8),
    scale_pos_weight=scale_pos,
    n_jobs=16, random_state=RANDOM_STATE,
    verbosity=0
)
final.fit(X, y)

pickle.dump(final, open(f'{MDL_OUT}/xgb_prob_clean_final.pkl', 'wb'))
with open(f'{MDL_OUT}/feature_list.json', 'w') as f:
    json.dump({
        'features': features,
        'model_code': 'PROB_XGB_002',
        'note': 'Clean 41-feature set from exp410 RFE — no point_id',
        'auc': auc,
        'avg_precision': ap,
        'f1': f1,
        'fold_aucs': fold_aucs,
        'n': len(y),
        'cv': 'spatial_block_50km',
    }, f, indent=2)

np.save(f'{RES_OUT}/xgb_oof_prob.npy', oof_prob)
np.save(f'{RES_OUT}/xgb_oof_true.npy', y)
print('Saved to ' + MDL_OUT)
print('DONE')
