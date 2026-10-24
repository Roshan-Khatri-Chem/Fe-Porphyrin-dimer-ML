import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Publication style (matches Ref. [1]) ──────────────────────────
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":           11,
    "axes.linewidth":      1.2,
    "xtick.direction":     "in",   "ytick.direction":    "in",
    "xtick.top":           True,   "ytick.right":        True,
    "xtick.major.size":    5,      "ytick.major.size":   5,
    "xtick.minor.size":    3,      "ytick.minor.size":   3,
    "xtick.major.width":   1.0,    "ytick.major.width":  1.0,
    "xtick.minor.visible": True,   "ytick.minor.visible":True,
    "legend.frameon":      False,
})

# ── Load data ─────────────────────────────────────────────────────
fes    = np.loadtxt("sum_hills/fes.dat")   # col 0: dist, col 1: FE [2]
pasted = np.loadtxt("pasted.dat")          # col 1: angle, col 3: dist,
                                           # col 5: PE              [3]
dist  = pasted[:, 3]
angle = pasted[:, 1]
pe    = pasted[:, 5]

# Sort HIGH energy first → low-energy points drawn LAST (on top)
order = np.argsort(pe)[::-1]
dist, angle, pe = dist[order], angle[order], pe[order]

# ── Figure ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(10, 4.2),
    gridspec_kw={"width_ratios": [1, 1.2]},
)

# ═══ (a) Free-energy profile ═════════════════════════════════════
ax1.plot(fes[:, 0], fes[:, 1], "k-", lw=1.4)
ax1.set_xlabel(r"$R_{\mathrm{Fe\text{-}Fe}}$ / nm", fontsize=12)
ax1.set_ylabel(r"Free Energy / kJ mol$^{-1}$",       fontsize=12)
ax1.set_xlim(0, 2.2)
ax1.set_ylim(-215, 10)
# Place (a) top-right so it avoids the repulsive wall
ax1.text(0.92, 0.95, "(a)", transform=ax1.transAxes,
         fontsize=14, fontweight="bold", va="top", ha="right",
         bbox=dict(facecolor="white", edgecolor="none",
                   alpha=0.8, pad=1.5))

# ═══ (b) Angle vs Fe–Fe distance, coloured by PE ════════════════
# Colour map: blue = low PE (stable), red = high PE
cmap = plt.cm.get_cmap("RdYlBu_r")
norm = mpl.colors.Normalize(vmin=100, vmax=550)

# Build RGBA with alpha fade: low PE → opaque, high PE → faint
rgba = cmap(norm(pe))
rgba[:, 3] = np.clip(1.0 - 0.55 * norm(pe), 0.25, 1.0)

ax2.scatter(dist, angle, c=rgba, s=5,
            edgecolors="none", rasterized=True)

# Dashed reference at 90° (T-shaped geometry)
ax2.axhline(90, color="grey", ls="--", lw=0.7, alpha=0.6)

ax2.set_xlabel(r"$R_{\mathrm{Fe\text{-}Fe}}$ / nm", fontsize=12)
ax2.set_ylabel(r"Angle / degrees",                   fontsize=12)
ax2.set_xlim(0, 1.6)
ax2.set_ylim(0, 180)
ax2.text(0.05, 0.95, "(b)", transform=ax2.transAxes,
         fontsize=14, fontweight="bold", va="top",
         bbox=dict(facecolor="white", edgecolor="none",
                   alpha=0.8, pad=1.5))

# Colorbar (ScalarMappable needed because c was raw RGBA)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax2, pad=0.02)
cbar.set_label(r"Potential Energy / kJ mol$^{-1}$", fontsize=11)

plt.tight_layout()
plt.savefig("figure_publication2.pdf", bbox_inches="tight")
plt.savefig("figure_publication2.png", dpi=300, bbox_inches="tight")
plt.show()
