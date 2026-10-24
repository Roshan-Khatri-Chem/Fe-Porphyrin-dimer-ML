import numpy as np
import matplotlib.pyplot as plt

# --- Publication style matching the PDF ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'legend.frameon': False,
})

# --- Load data ---
fes = np.loadtxt('sum_hills/fes.dat')        # cols: distance, free_energy, derivative
pasted = np.loadtxt('pasted.dat')             # cols: t, angle, t, dist, t, PE

# --- Create figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# ========== Panel (a): Free Energy Scan ==========
ax1.plot(fes[:, 0], fes[:, 1], color='black', linewidth=1.3)
ax1.set_xlabel(r'$R_{\mathrm{Fe\text{-}Fe}}$ / nm', fontsize=13)
ax1.set_ylabel(r'Free Energy / kJ mol$^{-1}$', fontsize=13)
ax1.set_xlim([0, 2.2])
ax1.set_ylim([-210, 5])
ax1.text(0.03, 0.95, '(a)', transform=ax1.transAxes,
         fontsize=14, fontweight='bold', va='top')

# ========== Panel (b): Angle vs Fe-Fe Distance ==========
# Note: your gnuplot cmd was 'u 2:6', but the figure axes show
#       x = distance (col 4), y = angle (col 2), color = PE (col 6)
sc = ax2.scatter(pasted[:, 3], pasted[:, 1],
                 c=pasted[:, 5], cmap='plasma',
                 s=3, alpha=0.8, edgecolors='none',
                 vmin=100, vmax=550, rasterized=True)
ax2.set_xlabel(r'$R_{\mathrm{Fe\text{-}Fe}}$ / nm', fontsize=13)
ax2.set_ylabel(r'Angle / degrees', fontsize=13)
ax2.set_xlim([0, 1.6])
ax2.set_ylim([0, 180])
ax2.text(0.03, 0.95, '(b)', transform=ax2.transAxes,
         fontsize=14, fontweight='bold', va='top', color='white')

cbar = fig.colorbar(sc, ax=ax2, pad=0.02)
cbar.set_label(r'Potential Energy / kJ mol$^{-1}$', fontsize=11)

plt.tight_layout()
plt.savefig('figure_publication.pdf', bbox_inches='tight')
plt.savefig('figure_publication.png', dpi=300, bbox_inches='tight')
plt.show()
