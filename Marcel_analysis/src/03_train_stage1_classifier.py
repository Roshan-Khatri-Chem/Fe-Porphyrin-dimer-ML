"""
Phase 5, Stage 1: Binary classifier for near-zero vs significant coupling.
"""

import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             classification_report)

sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
features = np.load(os.path.join(DATA_DIR, 'features.npy'))
labels_ev = np.load(os.path.join(DATA_DIR, 'labels_ev.npy'))
metadata = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))

# Binary labels: 1 = near-zero (coupling < 1e-4), 0 = significant
y = (labels_ev < 1e-4).astype(int)

# Train/test split from metadata
train_mask = metadata['split'] == 'train'
test_mask = metadata['split'] == 'test'

X_train, X_test = features[train_mask], features[test_mask]
y_train, y_test = y[train_mask.values], y[test_mask.values]

print(f"Train: {len(X_train)} ({y_train.sum()} near-zero, {(~y_train.astype(bool)).sum()} significant)")
print(f"Test:  {len(X_test)} ({y_test.sum()} near-zero, {(~y_test.astype(bool)).sum()} significant)")

# Pipeline: StandardScaler -> SVC
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True)),
])

param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 0.1, 0.5, 1.0, 2.0],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

print(f"\nBest parameters: {grid.best_params_}")
print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")

# Evaluate on test set
y_pred = grid.best_estimator_.predict(X_test)
y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred, target_names=['significant', 'near-zero'])

print(f"\nTest Confusion Matrix:")
print(f"               Pred Sig  Pred NZ")
print(f"  Actual Sig   {cm[0,0]:>6}    {cm[0,1]:>6}")
print(f"  Actual NZ    {cm[1,0]:>6}    {cm[1,1]:>6}")
print(f"\nTest ROC-AUC: {roc:.4f}")
print(f"\nClassification Report:\n{report}")

# Save model and scaler
joblib.dump(grid.best_estimator_, os.path.join(MODEL_DIR, 'stage1_classifier.pkl'))
# Also save the scaler separately
joblib.dump(grid.best_estimator_.named_steps['scaler'], os.path.join(MODEL_DIR, 'scaler_stage1.pkl'))
print("Saved: models/stage1_classifier.pkl, models/scaler_stage1.pkl")
