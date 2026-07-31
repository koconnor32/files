"""
recursive_feature_elimination.py
=================================
Recursive feature elimination for exp407 (0-20cm dropped, NWI binary 0 vs 1+2).
Best model: AUC=0.9740, AP=0.9242

Logic:
  1. Train 5-fold stratified CV with current feature set
  2. Record all metrics (AUC, AP, accuracy, precision, recall, F1)
  3. If ROC-AUC did NOT drop > 0.005 vs best seen:
       - Drop the single lowest mean feature importance
       - Repeat
  4. If ROC-AUC DID drop > 0.005 vs best seen: stop.

Outputs: 05_results/rfe/exp407/
"""

import os, json, pickle, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────
BASE          = '/scratch.global/ocon0444/peat_modeling'
CSV_PATH      = os.path.join(BASE, '00_data/processed/binary_peat_features_0_20_dropped.csv')
FEAT_JSON     = os.path.join(BASE, '03_models/exp407/feature_list.json')
OUT_DIR       = os.path.join(BASE, '05_results/rfe/exp407')
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_COL    = 'peat_binary'
RANDOM_STATE  = 42
N_FOLDS       = 5
N_TREES       = 200
AUC_TOLERANCE = 0.005

# ── LOAD DATA ─────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Shape: {df.shape}")

with open(FEAT_JSON) as f:
    feat_data = json.load(f)

feature_cols = feat_data['features']
print(f"  Starting features : {len(feature_cols)}")
print(f"  NWI type          : {feat_data.get('nwi_type', 'unknown')}")
print(f"  Dataset           : {feat_data.get('dataset', 'unknown')}")

# Drop NaN targets
valid = df[TARGET_COL].notna()
df    = df[valid].reset_index(drop=True)
y     = df[TARGET_COL].reset_index(drop=True)
print(f"  Training rows     : {len(y):,}  "
      f"(peat={int(y.sum()):,}  non={int((y==0).sum()):,})")


# ── HELPERS ───────────────────────────────────────────────────────
def prepare_X(df, cols):
    X = df[cols].apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    return X


def run_cv(df, y, feature_cols):
    X  = prepare_X(df, feature_cols)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics     = []
    fold_importances = np.zeros(len(feature_cols))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        rf = RandomForestClassifier(
            n_estimators=N_TREES,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        rf.fit(X_tr, y_tr)

        y_prob = rf.predict_proba(X_va)[:, 1]
        y_pred = rf.predict(X_va)

        fold_metrics.append({
            'fold':      fold,
            'roc_auc':   roc_auc_score(y_va, y_prob),
            'avg_prec':  average_precision_score(y_va, y_prob),
            'accuracy':  accuracy_score(y_va, y_pred),
            'precision': precision_score(y_va, y_pred, zero_division=0),
            'recall':    recall_score(y_va, y_pred, zero_division=0),
            'f1':        f1_score(y_va, y_pred, zero_division=0),
        })
        fold_importances += rf.feature_importances_

    fold_importances /= N_FOLDS
    fdf = pd.DataFrame(fold_metrics)

    metrics = {
        'mean_roc_auc':   round(fdf['roc_auc'].mean(),   4),
        'std_roc_auc':    round(fdf['roc_auc'].std(),    4),
        'mean_avg_prec':  round(fdf['avg_prec'].mean(),  4),
        'std_avg_prec':   round(fdf['avg_prec'].std(),   4),
        'mean_accuracy':  round(fdf['accuracy'].mean(),  4),
        'mean_precision': round(fdf['precision'].mean(), 4),
        'mean_recall':    round(fdf['recall'].mean(),    4),
        'mean_f1':        round(fdf['f1'].mean(),        4),
        'n_features':     len(feature_cols),
    }
    return metrics, fold_importances


# ── RECURSIVE ELIMINATION ─────────────────────────────────────────
current_features = list(feature_cols)
best_auc         = 0.0
best_round       = 0
best_features    = list(current_features)

round_results = []
dropped_log   = []

print(f"\n{'='*60}")
print(f"Starting RFE — {len(current_features)} features, "
      f"tolerance={AUC_TOLERANCE}")
print(f"{'='*60}\n")

grand_start = time.time()
rnd = 0

while len(current_features) > 1:
    t0 = time.time()
    rnd += 1

    print(f"Round {rnd:03d} | {len(current_features)} features")
    metrics, importances = run_cv(df, y, current_features)

    auc     = metrics['mean_roc_auc']
    elapsed = (time.time() - t0) / 60

    print(f"  AUC={auc:.4f} (best={best_auc:.4f})  "
          f"AP={metrics['mean_avg_prec']:.4f}  "
          f"Acc={metrics['mean_accuracy']:.4f}  "
          f"F1={metrics['mean_f1']:.4f}  "
          f"Rec={metrics['mean_recall']:.4f}  "
          f"elapsed={elapsed:.1f}min")

    row = {'round': rnd, 'elapsed_min': round(elapsed, 2), **metrics}
    round_results.append(row)

    # Update best
    if auc > best_auc:
        best_auc      = auc
        best_round    = rnd
        best_features = list(current_features)
        print(f"  *** New best AUC: {best_auc:.4f} at round {rnd} ***")

    # Stop if AUC dropped too much
    if rnd > 1 and (best_auc - auc) > AUC_TOLERANCE:
        print(f"\n  STOP: AUC dropped {best_auc - auc:.4f} > "
              f"tolerance {AUC_TOLERANCE}")
        print(f"  Best round was {best_round} with "
              f"{len(best_features)} features")
        break

    # Drop lowest importance feature
    imp_series = pd.Series(importances, index=current_features)
    drop_col   = imp_series.idxmin()
    drop_imp   = imp_series.min()

    dropped_log.append({
        'round':               rnd,
        'dropped_col':         drop_col,
        'importance':          round(float(drop_imp), 6),
        'auc_before':          auc,
        'n_features_before':   len(current_features),
    })

    print(f"  Dropping: {drop_col} (importance={drop_imp:.6f})")
    current_features.remove(drop_col)

    # Checkpoint every 10 rounds
    if rnd % 10 == 0:
        pd.DataFrame(round_results).to_csv(
            os.path.join(OUT_DIR, 'rfe_round_results.csv'), index=False)
        pd.DataFrame(dropped_log).to_csv(
            os.path.join(OUT_DIR, 'rfe_dropped_features.csv'), index=False)
        print(f"  [checkpoint saved at round {rnd}]")

total_min = (time.time() - grand_start) / 60

# ── SAVE RESULTS ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RFE complete in {total_min:.1f} min")
print(f"Best round : {best_round} | AUC : {best_auc:.4f} | "
      f"Features : {len(best_features)}")
print(f"{'='*60}")

round_df   = pd.DataFrame(round_results)
dropped_df = pd.DataFrame(dropped_log)

round_df.to_csv(os.path.join(OUT_DIR, 'rfe_round_results.csv'), index=False)
dropped_df.to_csv(os.path.join(OUT_DIR, 'rfe_dropped_features.csv'), index=False)

with open(os.path.join(OUT_DIR, 'rfe_best_features.json'), 'w') as f:
    json.dump({
        'exp_id':      'exp407',
        'best_round':  best_round,
        'best_auc':    best_auc,
        'n_features':  len(best_features),
        'features':    best_features,
        'tolerance':   AUC_TOLERANCE,
    }, f, indent=2)

with open(os.path.join(OUT_DIR, 'rfe_summary.json'), 'w') as f:
    json.dump({
        'exp_id':           'exp407',
        'source_features':  len(feature_cols),
        'best_round':       best_round,
        'best_auc':         best_auc,
        'best_n_features':  len(best_features),
        'best_features':    best_features,
        'total_rounds':     rnd,
        'total_min':        round(total_min, 2),
        'tolerance':        AUC_TOLERANCE,
        'dropped_order':    dropped_df['dropped_col'].tolist()
                            if len(dropped_df) > 0 else [],
    }, f, indent=2)

# Final importance on best feature set
print(f"\nTop 10 features in best model:")
X_best    = prepare_X(df, best_features)
cv        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=RANDOM_STATE)
imp_final = np.zeros(len(best_features))
for fold, (tr_idx, _) in enumerate(cv.split(X_best, y)):
    rf = RandomForestClassifier(n_estimators=N_TREES,
                                class_weight='balanced',
                                n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_best.iloc[tr_idx], y.iloc[tr_idx])
    imp_final += rf.feature_importances_
imp_final /= N_FOLDS

imp_df = pd.DataFrame({'feature': best_features, 'importance': imp_final})
imp_df = imp_df.sort_values('importance', ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, 'rfe_best_feature_importances.csv'),
              index=False)
print(imp_df.head(10).to_string(index=False))
print(f"\nOutputs: {OUT_DIR}")
