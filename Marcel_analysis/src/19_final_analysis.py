"""
v7: Final analysis — per-conformation table, publication figure,
and complete validation statistics.
"""

import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')

GAMMA_STAR_MEV = {
    1: 39.9246,  2:  0.1905,  3:  5.4858,  4:  5.5266,
    5: 42.1205,  6:  7.6328,  7: 22.4902,  8: 30.6074,
    9:  0.0000, 10:  0.0136, 11:  0.5987, 12:  0.3755,
   13:  0.0190, 14: 10.1553, 15:  0.0000,
}

GAMMA_MEV = {
    1: 0.0027,  2: 0.3755,  3: 0.4327,  4: 0.0435,
    5: 3.1157,  6: 3.1919,  7: 5.0096,  8: 0.9932,
    9: 0.1034, 10: 0.0027, 11: 0.0109, 12: 0.2259,
   13: 0.0163, 14: 0.0327, 15: 0.3510,
}

CATEGORIES = {
    1: 'distorted', 2: 'lateral-OOD', 3: 'in-dist', 4: 'in-dist',
    5: 'in-dist', 6: 'in-dist', 7: 'in-dist', 8: 'in-dist',
    9: 'geometric-node', 10: 'lateral-OOD', 11: 'in-dist', 12: 'in-dist',
    13: 'lateral-OOD', 14: 'distorted', 15: 'geometric-node',
}

# Approximate R_PD values from Table 1 of Bakenhaster et al.
# (parallel displacement in Angstrom)
# These are derived from the MD feature vectors: R_PD ≈ d_lateral
# For confs 9, 15: R_PD is at the interference minimum (~4.6 Å)

# ============================================================
# Part 1: Comprehensive per-conformation table
# ============================================================
print("=" * 70)
print("PART 1: Per-conformation analysis table")
print("=" * 70)

df = pd.read_csv(os.path.join(RESULTS_DIR, 'md_validation.csv'))
ok = df[df['status'] == 'OK'].copy().reset_index(drop=True)

ok['gamma_star_ref'] = ok['conf_id'].map(GAMMA_STAR_MEV)
ok['gamma_ref'] = ok['conf_id'].map(GAMMA_MEV)
ok['category'] = ok['conf_id'].map(CATEGORIES)

# Log ratios
ok['log_ratio_gamma'] = ok.apply(
    lambda r: np.log10(r['J_pred_mev'] / r['gamma_ref'])
    if r['gamma_ref'] > 0 and r['J_pred_mev'] > 0 else np.nan, axis=1)

ok['log_ratio_star'] = ok.apply(
    lambda r: np.log10(r['J_pred_mev'] / r['gamma_star_ref'])
    if r['gamma_star_ref'] > 0 and r['J_pred_mev'] > 0 else np.nan, axis=1)

ok['log_gamma_star_over_gamma'] = ok.apply(
    lambda r: np.log10(r['gamma_star_ref'] / r['gamma_ref'])
    if r['gamma_star_ref'] > 0 and r['gamma_ref'] > 0 else np.nan, axis=1)

ok['abs_log_ratio_star'] = ok['log_ratio_star'].abs()

# Save full table
ok.to_csv(os.path.join(RESULTS_DIR, 'final_validation_table.csv'), index=False)

# Print formatted
print(f"\n{'Conf':>4} {'Cat':>12} {'Γ*':>8} {'Γ':>8} {'Γ_pred':>8} "
      f"{'lr_Γ*':>7} {'lr_Γ':>7} {'Γ*/Γ':>6} {'dz':>5} {'d_lat':>5} {'cos4φ':>6}")
print("-" * 95)
for _, r in ok.iterrows():
    cid = int(r['conf_id'])
    lr_s = f"{r['log_ratio_star']:+.2f}" if pd.notna(r['log_ratio_star']) else "  N/A"
    lr_g = f"{r['log_ratio_gamma']:+.2f}" if pd.notna(r['log_ratio_gamma']) else "  N/A"
    gs_g = f"{r['log_gamma_star_over_gamma']:+.1f}" if pd.notna(r['log_gamma_star_over_gamma']) else " N/A"
    print(f"{cid:>4} {r['category']:>12} {r['gamma_star_ref']:>8.3f} {r['gamma_ref']:>8.4f} "
          f"{r['J_pred_mev']:>8.3f} {lr_s:>7} {lr_g:>7} {gs_g:>6} "
          f"{r['dz_ang']:>5.2f} {r['d_lateral']:>5.2f} {r['cos_4phi']:>6.3f}")

# ============================================================
# Part 2: Pearson r for subsets
# ============================================================
print("\n" + "=" * 70)
print("PART 2: Correlation analysis by subset")
print("=" * 70)

# All 13 valid (excl. geometric nodes)
valid_13 = ok[~ok['conf_id'].isin([9, 15])].copy()
r13, p13 = pearsonr(np.log10(valid_13['gamma_star_ref']),
                     np.log10(valid_13['J_pred_mev']))
mae_13 = valid_13['abs_log_ratio_star'].mean()
print(f"All 13 valid:         r = {r13:.3f}, p = {p13:.3f}, log MAE = {mae_13:.3f}")

# In-dist + distorted (excludes lateral-OOD and geometric-node)
in_dist_d = ok[ok['category'].isin(['in-dist', 'distorted'])].copy()
r_id, p_id = pearsonr(np.log10(in_dist_d['gamma_star_ref']),
                        np.log10(in_dist_d['J_pred_mev']))
mae_id = in_dist_d['abs_log_ratio_star'].mean()
print(f"In-dist + distorted (N={len(in_dist_d)}): r = {r_id:.3f}, p = {p_id:.3f}, log MAE = {mae_id:.3f}")

# In-dist only
in_dist_only = ok[ok['category'] == 'in-dist'].copy()
r_io, p_io = pearsonr(np.log10(in_dist_only['gamma_star_ref']),
                        np.log10(in_dist_only['J_pred_mev']))
mae_io = in_dist_only['abs_log_ratio_star'].mean()
print(f"In-dist only (N={len(in_dist_only)}): r = {r_io:.3f}, p = {p_io:.3f}, log MAE = {mae_io:.3f}")

# Lateral-OOD
lat_ood = ok[ok['category'] == 'lateral-OOD'].copy()
mae_lat = lat_ood['abs_log_ratio_star'].mean()
print(f"Lateral-OOD (N={len(lat_ood)}): log MAE = {mae_lat:.3f}")

# Within factor of 3
within_3 = (valid_13['abs_log_ratio_star'] < 0.5).sum()
within_10 = (valid_13['abs_log_ratio_star'] < 1.0).sum()
print(f"\nWithin factor of 3: {within_3}/13 ({within_3/13*100:.0f}%)")
print(f"Within factor of 10: {within_10}/13 ({within_10/13*100:.0f}%)")

# ============================================================
# Part 3: Publication figure (3 panels)
# ============================================================
print("\n" + "=" * 70)
print("PART 3: Publication figure")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

mpl.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

cat_style = {
    'in-dist':        {'color': '#1f77b4', 'marker': 'o', 'label': 'In-distribution'},
    'lateral-OOD':    {'color': '#ff7f0e', 'marker': '^', 'label': r'$R_\mathrm{PD}$ interference'},
    'distorted':      {'color': '#d62728', 'marker': 's', 'label': 'Intramolecular distortion'},
    'geometric-node': {'color': '#999999', 'marker': 'X', 'label': r'Geometric node ($\Gamma^*=0$)'},
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# --- Panel A: Parity plot predicted vs Gamma* ---
ax = axes[0]
lims = [5e-3, 80]

for cat, style in cat_style.items():
    subset = ok[ok['category'] == cat]
    for _, r in subset.iterrows():
        cid = int(r['conf_id'])
        pred = r['J_pred_mev']
        gs = r['gamma_star_ref']
        if cat == 'geometric-node':
            # Plot at left edge with downward arrow
            ax.scatter(lims[0] * 2, pred, c=style['color'], s=70,
                       marker=style['marker'], zorder=3, edgecolors='k', linewidths=0.5)
            ax.annotate(f"{cid}", (lims[0] * 2, pred),
                        xytext=(5, 3), textcoords='offset points', fontsize=7)
        else:
            ax.scatter(gs, pred, c=style['color'], s=70, marker=style['marker'],
                       zorder=3, edgecolors='k', linewidths=0.5)
            ax.annotate(f"{cid}", (gs, pred),
                        xytext=(4, 4), textcoords='offset points', fontsize=7)

ax.plot(lims, lims, 'k-', linewidth=1)
ax.fill_between(lims, [x / 3 for x in lims], [x * 3 for x in lims],
                alpha=0.1, color='grey')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r'$\Gamma^*_\mathrm{ref}$ (meV)')
ax.set_ylabel(r'$\Gamma_\mathrm{pred}$ (meV)')
ax.text(0.04, 0.96, f'log MAE = {mae_13:.2f}\n$r$ = {r13:.2f} (N=13)',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey'))

legend_handles = [Line2D([0], [0], marker=s['marker'], color='w',
                          markerfacecolor=s['color'], markersize=8,
                          markeredgecolor='k', markeredgewidth=0.5,
                          label=s['label']) for s in cat_style.values()]
ax.legend(handles=legend_handles, fontsize=7, loc='lower right')

# --- Panel B: Bar chart of |log ratio| per conformation ---
ax = axes[1]
confs = ok['conf_id'].values.astype(int)
abs_lr = []
bar_colors = []
for _, r in ok.iterrows():
    if pd.notna(r['log_ratio_star']):
        abs_lr.append(abs(r['log_ratio_star']))
    else:
        abs_lr.append(0)  # geometric nodes get 0 height
    bar_colors.append(cat_style[r['category']]['color'])

bars = ax.bar(range(len(confs)), abs_lr, color=bar_colors, edgecolor='k', linewidth=0.3)
ax.axhline(0.5, color='grey', linestyle='--', linewidth=1, label='Factor of 3')
ax.axhline(1.0, color='grey', linestyle=':', linewidth=0.8, label='Factor of 10')
ax.set_xticks(range(len(confs)))
ax.set_xticklabels(confs, fontsize=8)
ax.set_xlabel('Conformation')
ax.set_ylabel(r'$|\log_{10}(\Gamma_\mathrm{pred}/\Gamma^*_\mathrm{ref})|$')
ax.legend(fontsize=8, loc='upper right')

# Mark geometric nodes
for i, r in ok.iterrows():
    if r['category'] == 'geometric-node':
        ax.text(i, 0.05, r'$\Gamma^*\!=\!0$', ha='center', fontsize=6,
                color='#999999', style='italic')

# --- Panel C: R_PD vs log prediction error ---
ax = axes[2]
# Use d_lateral as proxy for R_PD
for _, r in ok.iterrows():
    cid = int(r['conf_id'])
    d_lat = r['d_lateral']
    lr = r['log_ratio_star'] if pd.notna(r['log_ratio_star']) else 0
    cat = r['category']
    style = cat_style[cat]
    ax.scatter(d_lat, lr, c=style['color'], s=70, marker=style['marker'],
               zorder=3, edgecolors='k', linewidths=0.5)
    ax.annotate(f"{cid}", (d_lat, lr), xytext=(4, 4),
                textcoords='offset points', fontsize=7)

ax.axhline(0, color='k', linewidth=0.8)
ax.axhline(0.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)
ax.axhline(-0.5, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)

# Mark approximate R_PD interference minima
for rpd_min, label in [(2.0, r'$R_\mathrm{PD}$ min 1'), (4.6, r'$R_\mathrm{PD}$ min 2')]:
    ax.axvline(rpd_min, color='#ff7f0e', linewidth=1.2, linestyle=':', alpha=0.6)
    ax.text(rpd_min + 0.1, ax.get_ylim()[1] if ax.get_ylim()[1] > 2 else 2.5,
            label, fontsize=7, color='#ff7f0e', rotation=90, va='top')

ax.set_xlabel(r'$d_\mathrm{lateral}$ (Å) [$\approx R_\mathrm{PD}$]')
ax.set_ylabel(r'$\log_{10}(\Gamma_\mathrm{pred}/\Gamma^*_\mathrm{ref})$')

plt.tight_layout()
for ext in ['png', 'pdf']:
    plt.savefig(os.path.join(FIG_DIR, f'pub_fig6_final.{ext}'))
plt.close()
print("Saved: pub_fig6_final.png/pdf")

# Also update the main validation figure (fig5)
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Panel A: vs Gamma (wrong reference, for comparison)
ax = axes[0]
lims_full = [5e-4, 80]
for _, r in ok.iterrows():
    if r['gamma_ref'] > 0 and r['J_pred_mev'] > 0:
        ax.scatter(r['gamma_ref'], r['J_pred_mev'], c='#999999', s=60,
                   zorder=3, edgecolors='k', linewidths=0.5)
        ax.annotate(f"{int(r['conf_id'])}", (r['gamma_ref'], r['J_pred_mev']),
                    xytext=(4, 4), textcoords='offset points', fontsize=7, color='#666')
ax.plot(lims_full, lims_full, 'k-', linewidth=1)
ax.fill_between(lims_full, [x/3 for x in lims_full], [x*3 for x in lims_full],
                alpha=0.1, color='grey')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims_full); ax.set_ylim(lims_full)
ax.set_xlabel(r'$\Gamma$ (meV) [CDFT-CI, full]')
ax.set_ylabel(r'$\Gamma_\mathrm{pred}$ (meV)')
ax.text(0.05, 0.95, f'vs $\\Gamma$ (incl. distortion)\nlog MAE = 1.93',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.7))

# Panel B: vs Gamma* (correct)
ax = axes[1]
for cat, style in cat_style.items():
    subset = ok[ok['category'] == cat]
    for _, r in subset.iterrows():
        cid = int(r['conf_id'])
        pred = r['J_pred_mev']
        gs = r['gamma_star_ref']
        if cat == 'geometric-node':
            ax.scatter(lims[0] * 2, pred, c=style['color'], s=70,
                       marker=style['marker'], zorder=3, edgecolors='k', linewidths=0.5)
            ax.annotate(f"{cid}", (lims[0] * 2, pred),
                        xytext=(5, 3), textcoords='offset points', fontsize=7)
        else:
            ax.scatter(gs, pred, c=style['color'], s=70, marker=style['marker'],
                       zorder=3, edgecolors='k', linewidths=0.5)
            ax.annotate(f"{cid}", (gs, pred),
                        xytext=(4, 4), textcoords='offset points', fontsize=7)

ax.plot(lims, lims, 'k-', linewidth=1)
ax.fill_between(lims, [x/3 for x in lims], [x*3 for x in lims],
                alpha=0.1, color='grey')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r'$\Gamma^*$ (meV) [geometry-only ref.]')
ax.set_ylabel(r'$\Gamma_\mathrm{pred}$ (meV)')
ax.text(0.04, 0.96, f'vs $\\Gamma^*$ (geometry only)\nlog MAE = {mae_13:.2f}, $r$ = {r13:.2f}',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.7))
ax.legend(handles=legend_handles, fontsize=7, loc='lower right')

plt.tight_layout()
for ext in ['png', 'pdf']:
    plt.savefig(os.path.join(FIG_DIR, f'pub_fig5_md_validation.{ext}'))
plt.close()
print("Saved: pub_fig5_md_validation.png/pdf (updated)")

# Save summary
summary = {
    'log_mae_all_13': float(mae_13),
    'pearson_r_all_13': float(r13), 'pearson_p_all_13': float(p13),
    'log_mae_in_dist_distorted': float(mae_id),
    'pearson_r_in_dist_distorted': float(r_id),
    'n_in_dist_distorted': int(len(in_dist_d)),
    'log_mae_in_dist_only': float(mae_io),
    'pearson_r_in_dist_only': float(r_io),
    'n_in_dist_only': int(len(in_dist_only)),
    'log_mae_lateral_ood': float(mae_lat),
    'within_factor_3': int(within_3),
    'within_factor_10': int(within_10),
}
with open(os.path.join(RESULTS_DIR, 'final_validation_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: results/final_validation_summary.json")
