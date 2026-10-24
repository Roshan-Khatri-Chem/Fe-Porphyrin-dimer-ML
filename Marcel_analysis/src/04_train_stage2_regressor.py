"""
Phase 5, Stage 2: Regressor on log10(coupling) for structures with coupling >= 1e-4 eV.
Train KRR, GPR, and RF. Compare and save all models.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
features = np.load(os.path.join(DATA_DIR, 'features.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
labels_log = np.load(os.path.join(DATA_DIR, 'labels_log.npy'))
metadata = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

# Filter to significant coupling (>= 1e-4 eV)
sig_mask = labels_ev >= 1e-4
X_all = features[sig_mask]
y_all = labels_log[sig_mask]
meta_sig = metadata[sig_mask].reset_index(drop=True)

train_mask = meta_sig['split'] == 'train'
test_mask = meta_sig['split'] == 'test'

X_train, X_test = X_all[train_mask.values], X_all[test_mask.values]
y_train, y_test = y_all[train_mask.values], y_all[test_mask.values]

print(f"Significant structures: {len(X_all)}")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Fit a shared scaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

results = {}

# ---- Model A: Kernel Ridge Regression ----
print("\n--- Model A: Kernel Ridge Regression ---")
krr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('krr', KernelRidge(kernel='rbf')),
])
krr_params = {
    'krr__alpha': [1e-4, 1e-3, 1e-2, 0.1, 1.0],
    'krr__gamma': [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0],
}
krr_grid = GridSearchCV(krr_pipe, krr_params, cv=5, scoring='neg_mean_absolute_error',
                        n_jobs=-1, verbose=1)
krr_grid.fit(X_train, y_train)
print(f"Best params: {krr_grid.best_params_}")
print(f"Best CV log MAE: {-krr_grid.best_score_:.4f}")
results['KRR'] = {
    'cv_log_mae': float(-krr_grid.best_score_),
    'best_params': {k: str(v) for k, v in krr_grid.best_params_.items()},
}
joblib.dump(krr_grid.best_estimator_, os.path.join(MODEL_DIR, 'stage2_krr.pkl'))

# ---- Model B: Gaussian Process Regression ----
print("\n--- Model B: Gaussian Process Regression ---")

# GPR needs scaled data (pipeline doesn't work well with GPR kernel optimization)
kernel_rbf = C(1.0) * RBF(length_scale=np.ones(8)) + WhiteKernel(1e-3)
kernel_matern = C(1.0) * Matern(length_scale=np.ones(8), nu=2.5) + WhiteKernel(1e-3)

gpr_rbf = GaussianProcessRegressor(kernel=kernel_rbf, n_restarts_optimizer=10,
                                    normalize_y=True, random_state=42)
gpr_matern = GaussianProcessRegressor(kernel=kernel_matern, n_restarts_optimizer=10,
                                       normalize_y=True, random_state=42)

print("Fitting GPR (RBF)...")
gpr_rbf.fit(X_train_s, y_train)
ll_rbf = gpr_rbf.log_marginal_likelihood_value_
print(f"  Log-likelihood (RBF): {ll_rbf:.4f}")

print("Fitting GPR (Matern)...")
gpr_matern.fit(X_train_s, y_train)
ll_matern = gpr_matern.log_marginal_likelihood_value_
print(f"  Log-likelihood (Matern): {ll_matern:.4f}")

# Select better kernel
if ll_matern > ll_rbf:
    gpr_best = gpr_matern
    gpr_kernel_name = 'Matern'
    ll_best = ll_matern
else:
    gpr_best = gpr_rbf
    gpr_kernel_name = 'RBF'
    ll_best = ll_rbf
print(f"  Selected: {gpr_kernel_name} (log-lik={ll_best:.4f})")

# CV score for GPR
gpr_cv_scores = cross_val_score(
    GaussianProcessRegressor(kernel=gpr_best.kernel_, n_restarts_optimizer=5,
                              normalize_y=True, random_state=42),
    X_train_s, y_train, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
)
gpr_cv_mae = -gpr_cv_scores.mean()
print(f"  CV log MAE: {gpr_cv_mae:.4f}")
results['GPR'] = {
    'cv_log_mae': float(gpr_cv_mae),
    'kernel': gpr_kernel_name,
    'log_likelihood': float(ll_best),
}
joblib.dump(gpr_best, os.path.join(MODEL_DIR, 'stage2_gpr.pkl'))

# ---- Model C: Random Forest ----
print("\n--- Model C: Random Forest ---")
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(random_state=42)),
])
rf_params = {
    'rf__n_estimators': [200, 500],
    'rf__max_features': ['sqrt', 0.5],
    'rf__min_samples_leaf': [1, 2, 3],
    'rf__max_depth': [None, 10, 20],
}
rf_search = RandomizedSearchCV(rf_pipe, rf_params, n_iter=30, cv=5,
                                scoring='neg_mean_absolute_error', n_jobs=-1,
                                random_state=42, verbose=1)
rf_search.fit(X_train, y_train)
print(f"Best params: {rf_search.best_params_}")
print(f"Best CV log MAE: {-rf_search.best_score_:.4f}")
results['RF'] = {
    'cv_log_mae': float(-rf_search.best_score_),
    'best_params': {k: str(v) for k, v in rf_search.best_params_.items()},
}

# Save RF feature importances
rf_model = rf_search.best_estimator_.named_steps['rf']
importances = rf_model.feature_importances_
feature_names = ['dz', 'dx', 'dy', 'cos_theta', 'cos_4phi', 'sin_4phi', 'cos_4psi', 'sin_4psi']
print("\nRF Feature Importances:")
for name, imp in zip(feature_names, importances):
    print(f"  {name:<12}: {imp:.4f}")

joblib.dump(rf_search.best_estimator_, os.path.join(MODEL_DIR, 'stage2_rf.pkl'))

# Save shared scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_stage2.pkl'))

# ---- Comparison ----
print("\n" + "=" * 60)
print("MODEL COMPARISON (CV log MAE — lower is better)")
print("=" * 60)
print(f"{'Model':<10} {'CV log MAE':>12}")
print("-" * 24)
for name in ['KRR', 'GPR', 'RF']:
    print(f"{name:<10} {results[name]['cv_log_mae']:>12.4f}")

best_model = min(results, key=lambda k: results[k]['cv_log_mae'])
print(f"\nBest model: {best_model}")

# Save comparison
with open(os.path.join(RESULTS_DIR, 'cv_comparison.json'), 'w') as f:
    json.dump(results, f, indent=2)
print("Saved: results/cv_comparison.json")
print("Saved: models/stage2_krr.pkl, stage2_gpr.pkl, stage2_rf.pkl, scaler_stage2.pkl")
