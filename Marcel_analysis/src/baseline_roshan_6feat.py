"""Reproduce the Roshan KRR baseline using his actual 6-feature descriptor.

From his slides (data/from_roshan/Porphyrin_dimer_KRR_results.pdf):
  Features (6):
    - Interlayer_Dist_Ang     (perpendicular distance between porphyrin planes)
    - FeFe_Dist_Ang           (Fe-Fe distance)
    - NN_Dist_1_Ang           (nearest N-N distance between the two monomers)
    - Lateral_Slippage_Ang    (in-plane displacement magnitude)
    - Theta (degrees)         (angle between porphyrin normals)
    - Psi   (degrees)         (azimuthal angle of the lateral slippage direction)
  Filter: couplings >= 1e-4 eV (his 597-structure subset)
  Model:  KRR(kernel='rbf', alpha=0.01, gamma=1.0)
  Eval:   stratified 4:1 train/test, 10-fold CV on training set
  Reported: Test MAE = 0.00690 eV, Test R^2 = 0.863
"""
import os, sys, json
import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
ROOT = os.path.join(THIS, '..')
DATA = os.path.join(ROOT, 'data', 'processed')
RESULTS = os.path.join(ROOT, 'results')

from geometry import load_dimer_xyz, monomer_frame


def features_roshan_6(m1_pos, m1_sym, m2_pos, m2_sym):
    """Reproduce Roshan's 6 features.
       1. Interlayer_Dist_Ang  - perpendicular distance between porphyrin planes
       2. FeFe_Dist_Ang        - Fe-Fe Euclidean distance
       3. NN_Dist_1_Ang        - nearest N-N distance between the two monomer N sets
       4. Lateral_Slippage_Ang - sqrt(d_x^2 + d_y^2) in monomer-1 frame
       5. Theta (degrees)      - angle between porphyrin normals (0..180)
       6. Psi   (degrees)      - azimuthal angle of the lateral slip direction
    """
    m1_pos = np.asarray(m1_pos); m1_sym = np.asarray(m1_sym)
    m2_pos = np.asarray(m2_pos); m2_sym = np.asarray(m2_sym)

    fe1 = m1_pos[m1_sym == 'Fe'][0]
    fe2 = m2_pos[m2_sym == 'Fe'][0]
    feFe = float(np.linalg.norm(fe1 - fe2))

    c1, n1, x1 = monomer_frame(m1_pos, m1_sym)
    c2, n2, _  = monomer_frame(m2_pos, m2_sym)

    r_vec = c2 - c1
    n1_or = n1 * np.sign(np.dot(n1, r_vec))    # orient toward m2
    interlayer = float(abs(np.dot(r_vec, n1_or)))

    # Lateral slippage: project r_vec onto the m1 plane, take magnitude
    r_in_plane = r_vec - np.dot(r_vec, n1_or) * n1_or
    lateral = float(np.linalg.norm(r_in_plane))

    # Theta: angle between normals
    cos_t = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(abs(cos_t))))

    # Psi: azimuthal direction of the slip vector in m1's local frame
    y1 = np.cross(n1_or, x1)
    dx = float(np.dot(r_in_plane, x1))
    dy = float(np.dot(r_in_plane, y1))
    psi_deg = float(np.degrees(np.arctan2(dy, dx)))

    # NN_Dist_1: minimum N-N distance between the 4 core N atoms of each monomer
    N1_all = m1_pos[m1_sym == 'N']
    N2_all = m2_pos[m2_sym == 'N']
    N1 = N1_all[np.argsort(np.linalg.norm(N1_all - fe1, axis=1))[:4]]
    N2 = N2_all[np.argsort(np.linalg.norm(N2_all - fe2, axis=1))[:4]]
    NN_min = float(np.min([np.linalg.norm(a - b) for a in N1 for b in N2]))

    return np.array([interlayer, feFe, NN_min, lateral, theta_deg, psi_deg], dtype=float)


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'master_dataset.csv'))
    df_704 = df[df['dataset'] == '704_systematic'].copy().reset_index(drop=True)

    print('Computing Roshan 6-D features for 704 systematic...')
    rows = []
    for _, r in df_704.iterrows():
        xyz = os.path.join(ROOT, 'data', 'structures', r['name'] + '.xyz')
        if not os.path.exists(xyz): continue
        try:
            d = load_dimer_xyz(xyz)
            f = features_roshan_6(d['monomer1']['positions'], d['monomer1']['symbols'],
                                  d['monomer2']['positions'], d['monomer2']['symbols'])
            rows.append((r['name'], r['split'], r['J_cdft_eV'], f))
        except Exception as e:
            print(f'  failed {r["name"]}: {e}')

    X = np.array([row[3] for row in rows])
    J = np.array([row[2] for row in rows])
    splits = np.array([row[1] for row in rows])

    valid = J >= 1e-4
    train = (splits == 'train') & valid
    test  = (splits == 'test')  & valid
    print(f'  Filter J >= 1e-4: train={train.sum()}, test={test.sum()}, total={valid.sum()}')

    # ---- Roshan's exact configuration ----
    print('\n=== Roshan-style KRR: 6 features, J>=1e-4, alpha=0.01, gamma=1.0, RBF, linear target ===')
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('krr', KernelRidge(kernel='rbf', alpha=0.01, gamma=1.0)),
    ])
    pipe.fit(X[train], J[train])

    yp_train = pipe.predict(X[train])
    yp_test  = pipe.predict(X[test])

    print(f'  Train MAE = {mean_absolute_error(J[train], yp_train):.5f}')
    print(f'  Train R^2 = {r2_score(J[train], yp_train):.4f}')
    print(f'  Test  MAE = {mean_absolute_error(J[test],  yp_test) :.5f}')
    print(f'  Test  R^2 = {r2_score(J[test],  yp_test) :.4f}')
    print(f'  Roshan reported: Train MAE 0.00326, Train R^2 0.880, Test MAE 0.00690, Test R^2 0.863')

    # ---- 10-fold CV-on-train, evaluate on fixed test set (his protocol) ----
    print('\n=== Roshan-style 10-fold-on-train evaluation on fixed test set ===')
    from sklearn.model_selection import KFold
    train_idx = np.where(train)[0]
    test_idx  = np.where(test)[0]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    test_r2s = []
    test_maes = []
    for tr_fold, _ in kf.split(train_idx):
        pipe_fold = Pipeline([
            ('scaler', StandardScaler()),
            ('krr', KernelRidge(kernel='rbf', alpha=0.01, gamma=1.0)),
        ])
        idx_fold = train_idx[tr_fold]
        pipe_fold.fit(X[idx_fold], J[idx_fold])
        yp = pipe_fold.predict(X[test_idx])
        test_r2s.append(r2_score(J[test_idx], yp))
        test_maes.append(mean_absolute_error(J[test_idx], yp))
    print(f'  Test R^2 across 10 folds: mean = {np.mean(test_r2s):.4f} +- {np.std(test_r2s):.4f}')
    print(f'  Test MAE across 10 folds: mean = {np.mean(test_maes):.5f} +- {np.std(test_maes):.5f}')
    print(f'  Roshan reported: 0.85825 +- 0.01858, MAE 0.00700 +- 0.00051')

    # ---- Same evaluation but with our v4_srsh_aug feature set ----
    print('\n=== Apples-to-apples: our 14-feature v4_srsh_aug, same train/test, KRR ===')
    GEOM = ['dz','dx','dy','cos_theta','cos_4phi','sin_4phi',
            'cos_4psi','sin_4psi','d_lateral','cos_4alpha','sin_4alpha']
    df_aligned = df_704.iloc[[df_704.index[df_704['name']==r[0]][0] for r in rows]].reset_index(drop=True)
    X11 = df_aligned[GEOM].values
    s12 = df_aligned['S12_srsh_cation'].values
    X14 = np.column_stack([X11, s12, s12**2, np.log10(np.abs(s12)+1e-12)])
    finite = np.isfinite(s12)
    train_v4 = train & finite
    test_v4  = test & finite

    # KRR on 14 features (linear target, hyperparam search)
    from sklearn.model_selection import GridSearchCV
    pipe_krr14 = Pipeline([('s', StandardScaler()), ('krr', KernelRidge(kernel='rbf'))])
    grid = GridSearchCV(pipe_krr14,
        {'krr__alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
         'krr__gamma': [0.01, 0.1, 1.0, 10.0]},
        cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X14[train_v4], J[train_v4])
    best14 = grid.best_estimator_
    yp14 = best14.predict(X14[test_v4])
    print(f'  14-feat KRR (best CV params {grid.best_params_}):')
    print(f'    Test MAE = {mean_absolute_error(J[test_v4], yp14):.5f}')
    print(f'    Test R^2 = {r2_score(J[test_v4], yp14):.4f}')

    # RF on 14 features (log target — our actual production model)
    print('\n=== Our v4_srsh_aug RF (log target) on the same test set ===')
    from sklearn.ensemble import RandomForestRegressor
    pipe_rf = Pipeline([('s', StandardScaler()),
                        ('rf', RandomForestRegressor(n_estimators=500, max_features=0.5,
                                                     max_depth=20, random_state=42, n_jobs=-1))])
    log_J = np.log10(J)
    pipe_rf.fit(X14[train_v4], log_J[train_v4])
    yp_log = pipe_rf.predict(X14[test_v4])
    yp_lin = 10**yp_log
    print(f'  Test Lin-MAE = {mean_absolute_error(J[test_v4], yp_lin):.5f}')
    print(f'  Test Lin-R^2 = {r2_score(J[test_v4], yp_lin):.4f}')
    print(f'  Test Log-MAE = {mean_absolute_error(log_J[test_v4], yp_log):.4f}')
    print(f'  Test Log-R^2 = {r2_score(log_J[test_v4], yp_log):.4f}')

    out = {
        'roshan_6feat_single_split': {
            'train_mae': float(mean_absolute_error(J[train], yp_train)),
            'train_r2':  float(r2_score(J[train], yp_train)),
            'test_mae':  float(mean_absolute_error(J[test], yp_test)),
            'test_r2':   float(r2_score(J[test], yp_test)),
        },
        'roshan_6feat_10fold_on_train': {
            'test_r2_mean':  float(np.mean(test_r2s)),
            'test_r2_std':   float(np.std(test_r2s)),
            'test_mae_mean': float(np.mean(test_maes)),
            'test_mae_std':  float(np.std(test_maes)),
        },
        'reported_in_slides': {
            'train_mae': 0.00326, 'train_r2': 0.880,
            'test_mae':  0.00690, 'test_r2': 0.863,
            'test_r2_10fold_mean': 0.85825, 'test_r2_10fold_std': 0.01858,
        },
        'our_14feat_KRR_linear_target': {
            'test_mae': float(mean_absolute_error(J[test_v4], yp14)),
            'test_r2':  float(r2_score(J[test_v4], yp14)),
        },
        'our_14feat_RF_log_target': {
            'test_lin_mae': float(mean_absolute_error(J[test_v4], yp_lin)),
            'test_lin_r2':  float(r2_score(J[test_v4], yp_lin)),
            'test_log_mae': float(mean_absolute_error(log_J[test_v4], yp_log)),
            'test_log_r2':  float(r2_score(log_J[test_v4], yp_log)),
        },
        'protocol_notes': {
            'features': 'Roshan: 6 features (Interlayer, FeFe, NN_min, Lat_Slip, Theta_deg, Psi_deg)',
            'filter':   'J >= 1e-4 eV (yields 597 structures)',
            'split':    'stratified 4:1 train/test (kept fixed)',
            'KRR_hyperparams': 'alpha=0.01, gamma=1.0, RBF',
            'CV':       '10-fold on training set, each fold-model evaluated on fixed test set',
        },
    }
    out_path = os.path.join(RESULTS, 'baseline_roshan_6feat.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
