import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib import rcParams
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NearestNDInterpolator
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── Typography ───────────────────────────────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Palatino Linotype', 'Palatino', 'Georgia', 'DejaVu Serif']
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'

# ─── Load real data ───────────────────────────────────────────────────────────
df_raw = pd.read_csv('.\plot1_strategy\penetration_strategy_data.csv')
df_raw['P_AV_pct']  = (df_raw['P_AV']  * 100).round(1)
df_raw['P_CHV_pct'] = (df_raw['P_CHV'] * 100).round(1)
df_raw['P_CAV_pct'] = (df_raw['P_CAV'] * 100).round(1)

# ─── Design ───────────────────────────────────────────────────────────────────
strategy_colors = {
    'GL':   '#95A5A6',
    'NCAL': '#3498DB',
    'CL':   '#F39C12',
    'AL':   '#E74C3C',
    'CAL':  '#27AE60',
}
STRAT_INT = {'GL': 0, 'CL': 1, 'NCAL': 2, 'AL': 3, 'CAL': 4}
INT_STRAT  = {v: k for k, v in STRAT_INT.items()}

strategy_order_legend = ['GL', 'AL', 'CL', 'NCAL', 'CAL']
DARK = '#1C2833'
MID  = '#2C3E50'

SLICES    = [0, 20, 40, 60, 80]
Z_SPACING = 400   # tall vertical gap between slices

# ─── Build smooth RGBA for one P_CAV slice ───────────────────────────────────
# Domain: P_AV ∈ [0, 100-pcav], P_CHV ∈ [0, 100-pcav], P_AV + P_CHV ≤ 100-pcav
# (triangular domain)

GRID_RES = 350
SIGMA    = 100.0

def build_slice_rgba(p_cav_pct):
    p_cav_frac = p_cav_pct / 100.0
    sub = df_raw[np.isclose(df_raw['P_CAV'], p_cav_frac, atol=1e-6)].copy()

    max_val = 100.0 - p_cav_pct   # max P_AV or P_CHV in %

    # Fine grid
    gx = np.linspace(0, max_val, GRID_RES)   # P_AV
    gy = np.linspace(0, max_val, GRID_RES)   # P_CHV
    GX, GY = np.meshgrid(gx, gy)

    # Triangular domain mask: P_AV + P_CHV <= max_val
    valid = (GX + GY) <= (max_val + 0.5)

    # Nearest-neighbour interpolation from real data
    pts  = sub[['P_AV_pct', 'P_CHV_pct']].values
    labs = sub['Strategy'].map(STRAT_INT).values.astype(float)
    interp = NearestNDInterpolator(pts, labs)

    # Only interpolate valid region
    raw = np.full((GRID_RES, GRID_RES), -1.0)
    flat_x = GX[valid].ravel()
    flat_y = GY[valid].ravel()
    raw_vals = interp(flat_x, flat_y)
    raw[valid] = raw_vals

    # Gaussian smooth per-strategy probability → clean boundaries
    n = len(STRAT_INT)
    probs = np.zeros((n, GRID_RES, GRID_RES))
    for si in range(n):
        mask = (raw == si).astype(float)
        probs[si] = gaussian_filter(mask, sigma=SIGMA)

    smoothed = np.argmax(probs, axis=0)

    # Build RGBA — only in valid domain
    rgba = np.zeros((GRID_RES, GRID_RES, 4))
    for si, sname in INT_STRAT.items():
        m = (smoothed == si) & valid
        rgba[m] = mcolors.to_rgba(strategy_colors[sname], alpha=0.92)

    return gx, gy, rgba, max_val


# ─── Convert RGBA image → 3D quad collection ─────────────────────────────────
DOWN = 3   # downsample block size

def draw_slice(ax, p_cav_pct, z_disp, slice_idx):
    gx, gy, rgba, max_val = build_slice_rgba(p_cav_pct)

    polys   = []
    fcolors = []

    for ri in range(0, GRID_RES, DOWN):
        for ci in range(0, GRID_RES, DOWN):
            r2 = min(ri + DOWN, GRID_RES)
            c2 = min(ci + DOWN, GRID_RES)
            block = rgba[ri:r2, ci:c2]
            if block[:, :, 3].mean() < 0.05:
                continue
            col = block.mean(axis=(0, 1))
            x0 = gx[ci]
            x1 = gx[min(ci + DOWN, GRID_RES - 1)]
            y0 = gy[ri]
            y1 = gy[min(ri + DOWN, GRID_RES - 1)]
            verts = [
                [x0, y0, z_disp],
                [x1, y0, z_disp],
                [x1, y1, z_disp],
                [x0, y1, z_disp],
            ]
            polys.append(verts)
            fcolors.append(tuple(col))

    if polys:
        pc = Poly3DCollection(
            polys,
            facecolors=fcolors,
            edgecolors='none',
            linewidths=0,
            zorder=slice_idx * 6 + 2,
            rasterized=True,
        )
        ax.add_collection3d(pc)

    # ── Clean triangular domain border (3 sides: bottom, right diagonal, left)
    mv = max_val
    # Bottom edge: P_CHV=0, P_AV: 0→mv
    # Hypotenuse: P_AV+P_CHV=mv (from (mv,0) to (0,mv))
    # Left edge: P_AV=0, P_CHV: mv→0
    n_pts = 80
    bx = np.concatenate([
        np.linspace(0,  mv, n_pts),   # bottom
        np.linspace(mv, 0,  n_pts),   # hypotenuse
        np.zeros(n_pts),               # left edge
        [0]
    ])
    by = np.concatenate([
        np.zeros(n_pts),               # bottom
        np.linspace(0,  mv, n_pts),   # hypotenuse
        np.linspace(mv, 0,  n_pts),   # left edge
        [0]
    ])
    bz = np.full_like(bx, z_disp)
    ax.plot(bx, by, bz, color='#2C3E50', lw=1.8, alpha=0.90,
            zorder=slice_idx * 6 + 8)

    # ── XY axis tick labels on each slice ──────────────────────────────────────
    tick_vals = [0, 20, 40, 60, 80, 100]
    for tv in tick_vals:
        if tv <= mv - 1:
            ax.text(tv, -6, z_disp, f'{tv}', fontsize=10, color=MID,
                    ha='center', va='top', zorder=slice_idx * 6 + 9)
            ax.text(-6, tv, z_disp, f'{tv}', fontsize=10, color=MID,
                    ha='right', va='center', zorder=slice_idx * 6 + 9)

    # ── Light interior grid lines (clipped to triangle)
    for gv in [20, 40, 60]:
        if gv < mv - 1:
            # Vertical: P_AV=gv, P_CHV from 0 to (mv-gv)
            ax.plot([gv, gv], [0, mv - gv], [z_disp, z_disp],
                    color='white', lw=0.55, alpha=0.45,
                    zorder=slice_idx * 6 + 4)
            # Horizontal: P_CHV=gv, P_AV from 0 to (mv-gv)
            ax.plot([0, mv - gv], [gv, gv], [z_disp, z_disp],
                    color='white', lw=0.55, alpha=0.45,
                    zorder=slice_idx * 6 + 4)

    # ── Slice label at the hypotenuse midpoint (top of triangle when viewed in 3D)
    # The "top" visible corner is where P_AV=0, P_CHV=mv (left back vertex)
    ax.text(mv + 1, -1, z_disp,
            f'$P_{{\\mathrm{{CAV}}}}={p_cav_pct:.0f}\\%$',
            fontsize=13, fontweight='bold', color=MID,
            ha='left', va='top', zorder=300)


# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 16), facecolor='white', dpi=300)
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#F4F6F7')

for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#BFC9CA')
ax.grid(False)

for slice_idx, p_cav in enumerate(SLICES):
    z_disp = slice_idx * Z_SPACING
    print(f'  Rendering P_CAV={p_cav}%...')
    draw_slice(ax, p_cav, z_disp, slice_idx)

# ─── Vertical connector at (0,0) corner ──────────────────────────────────────
ax.plot([0, 0], [0, 0],
        [0, (len(SLICES)-1)*Z_SPACING],
        color='#7F8C8D', lw=0.9, ls=':', alpha=0.55, zorder=1)

# # ─── Transition arrows ────────────────────────────────────────────────────────
# arrow_data = [
#     (0, 1, 'CL, NCAL emerge\nat $P_\\mathrm{CAV}=20\\%$', 45, 30),
#     (1, 2, 'CAL emerges\nat $P_\\mathrm{CAV}=40\\%$',     38, 22),
# ]
# for from_i, to_i, label, ta, tc in arrow_data:
#     z0 = from_i * Z_SPACING + 10
#     z1 = to_i   * Z_SPACING - 10
#     if z1 > z0:
#         ax.quiver(ta, tc, z0, 0, 0, z1 - z0,
#                   color='#1A3A5C', linewidth=1.5,
#                   arrow_length_ratio=0.25, alpha=0.88, zorder=90)
#         ax.text(ta + 2, tc, (z0 + z1)/2, label,
#                 fontsize=11, color='#1A3A5C', style='italic',
#                 ha='left', va='center', zorder=100)

# ─── Axes ─────────────────────────────────────────────────────────────────────
z_ticks  = [i * Z_SPACING for i in range(len(SLICES))]
z_labels = [f'{p}%' for p in SLICES]

# ax.set_xlim(-10, 112)
# ax.set_ylim(-10, 112)
# ax.set_zlim(-20, (len(SLICES)-1)*Z_SPACING + 35)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(0, (len(SLICES)-1)*Z_SPACING)

ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_zticks(z_ticks)
ax.set_zticklabels(z_labels)

lkw = dict(fontsize=14, fontweight='bold', color=DARK)
ax.set_xlabel(r'AV Penetration  $P_\mathrm{AV}$ / %',   labelpad=14, **lkw)
ax.set_ylabel(r'CHV Penetration  $P_\mathrm{CHV}$ / %', labelpad=14, **lkw)
ax.tick_params(axis='x', labelsize=12,  colors=MID, pad=2)
ax.tick_params(axis='y', labelsize=12,  colors=MID, pad=2)
ax.tick_params(axis='z', labelsize=13, colors=MID, pad=4)
ax.view_init(elev=28, azim=222)

# ─── Legend ───────────────────────────────────────────────────────────────────
legend_labels = {
    'GL':   'GL — General Lane',
    'AL':   'AL — Autonomous Lane',
    'CL':   'CL — Connected Lane',
    'NCAL': 'NCAL — Non-CAV Autonomous Lane',
    'CAL':  'CAL — Connected Autonomous Lane',
}
handles = [
    mpatches.Patch(facecolor=strategy_colors[s], edgecolor='#5D6D7E',
                   label=legend_labels[s], linewidth=0.8, alpha=0.93)
    for s in strategy_order_legend
]
leg = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 1.0),
                fontsize=12, title='Optimal Lane Strategy', title_fontsize=13.5,
                framealpha=0.97, edgecolor='#BFC9CA', fancybox=False,
                frameon=True, handlelength=1.5, handleheight=1.2)
leg.get_title().set_fontweight('bold')
leg.get_title().set_color(DARK)

# ─── Titles ───────────────────────────────────────────────────────────────────
fig.text(0.52, 0.97,
         r'Optimal Strategy Transition Across $P_\mathrm{CAV}$ Levels — Layered Slice View',
         ha='center', va='top', fontsize=18, fontweight='bold', color=DARK)
fig.text(0.52, 0.934,
         r'Each slice = one $P_\mathrm{CAV}$ level  $\vert$  '
         r'Filled area = dominant optimal lane strategy  $\vert$  '
         r'Shrinking triangular domain: $P_\mathrm{AV}+P_\mathrm{CHV} \leq 1-P_\mathrm{CAV}$  $\vert$  '
         r'Arrows = critical transition thresholds',
         ha='center', va='top', fontsize=12, color='#566573', style='italic')

plt.subplots_adjust(left=0.04, right=0.97, top=0.91, bottom=0.05)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = f'.\\plot1_strategy\\fig_real_final_{timestamp}.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}')