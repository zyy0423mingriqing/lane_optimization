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

# ─── Load data ────────────────────────────────────────────────────────────────
df_raw = pd.read_csv('./plot1_strategy/penetration_strategy_data.csv')
df_raw['P_AV_pct']  = (df_raw['P_AV']  * 100).round(1)
df_raw['P_CHV_pct'] = (df_raw['P_CHV'] * 100).round(1)
df_raw['P_CAV_pct'] = (df_raw['P_CAV'] * 100).round(1)

# ─── Colors & mappings ────────────────────────────────────────────────────────
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

DARK         = '#1C2833'
MID          = '#2C3E50'
BORDER_COLOR = '#1C2833'
BORDER_LW    = 2.0
GRID_LW      = 0.6

SLICES = [0, 20, 40, 60, 80]

# ─── KEY: Non-uniform Z positions matching fig2 visual style ─────────────────
# Fig2 has HUGE gap between bottom two layers, then progressively tighter.
# The critical insight: bottom gap ~2x the top gap.
# P_CAV=0%  at z=0 (bottom, largest layer)
# P_CAV=80% at z=top (smallest layer)
Z_POSITIONS = [
    0,     # P_CAV = 0%   (bottom)
    1800,  # P_CAV = 20%  — large gap from bottom
    3600,  # P_CAV = 40%  — medium gap
    5400,  # P_CAV = 60%  — smaller gap
    7200,  # P_CAV = 80%  — smallest gap (top)
]
# Gaps: 1800, 1400, 1100, 800  → clearly decreasing, like fig2

Z_MAX = max(Z_POSITIONS)

# ─── Font sizes — ALL large ───────────────────────────────────────────────────
# FS_TICK_ON_SLICE = 16    # numbers printed ON each slice (close to triangle)
# FS_SLICE_LABEL   = 21    # "P_CAV = X%" on each slice
# FS_AXIS_LABEL    = 21    # x/y/z axis titles
# FS_OUTER_TICKS   = 17    # outer frame tick labels
# FS_Z_TICKS       = 18    # z-axis tick labels
# FS_LEGEND        = 18
# FS_LEGEND_TITLE  = 20
# FS_TITLE         = 27
# FS_SUBTITLE      = 15
# FS_ARROW_LABEL   = 15
FS_TICK_ON_SLICE = 25    # numbers printed ON each slice (close to triangle)
FS_SLICE_LABEL   = 30    # "P_CAV = X%" on each slice
FS_AXIS_LABEL    = 25    # x/y/z axis titles
FS_OUTER_TICKS   = 30    # outer frame tick labels
FS_Z_TICKS       = 30    # z-axis tick labels
FS_LEGEND        = 30
FS_LEGEND_TITLE  = 30
FS_TITLE         = 30
FS_SUBTITLE      = 30
FS_ARROW_LABEL   = 30

# How close tick numbers sit to the triangle edge (smaller = closer)
TICK_OFFSET = 3   # was 10 in original, now very tight

# ─── Build smooth RGBA for one P_CAV slice ───────────────────────────────────
GRID_RES = 350
SIGMA    = 50.0

def build_slice_rgba(p_cav_pct):
    p_cav_frac = p_cav_pct / 100.0
    sub = df_raw[np.isclose(df_raw['P_CAV'], p_cav_frac, atol=1e-6)].copy()
    max_val = 100.0 - p_cav_pct

    gx = np.linspace(0, max_val, GRID_RES)
    gy = np.linspace(0, max_val, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    valid = (GX + GY) <= (max_val + 0.5)

    pts  = sub[['P_AV_pct', 'P_CHV_pct']].values
    labs = sub['Strategy'].map(STRAT_INT).values.astype(float)
    interp = NearestNDInterpolator(pts, labs)

    raw = np.full((GRID_RES, GRID_RES), -1.0)
    flat_x = GX[valid].ravel()
    flat_y = GY[valid].ravel()
    raw[valid] = interp(flat_x, flat_y)

    n = len(STRAT_INT)
    probs = np.zeros((n, GRID_RES, GRID_RES))
    for si in range(n):
        mask = (raw == si).astype(float)
        probs[si] = gaussian_filter(mask, sigma=SIGMA)

    smoothed = np.argmax(probs, axis=0)

    rgba = np.zeros((GRID_RES, GRID_RES, 4))
    for si, sname in INT_STRAT.items():
        m = (smoothed == si) & valid
        rgba[m] = mcolors.to_rgba(strategy_colors[sname], alpha=0.93)

    return gx, gy, rgba, max_val


DOWN = 3  # downsample block size for Poly3DCollection

def draw_slice(ax, p_cav_pct, z_disp, slice_idx):
    gx, gy, rgba, max_val = build_slice_rgba(p_cav_pct)
    mv = max_val

    # ── Filled colored quads ──────────────────────────────────────────────────
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
            x0 = gx[ci];  x1 = gx[min(ci + DOWN, GRID_RES - 1)]
            y0 = gy[ri];  y1 = gy[min(ri + DOWN, GRID_RES - 1)]
            polys.append([[x0,y0,z_disp],[x1,y0,z_disp],[x1,y1,z_disp],[x0,y1,z_disp]])
            fcolors.append(tuple(col))

    if polys:
        pc = Poly3DCollection(polys, facecolors=fcolors, edgecolors='none',
                              linewidths=0, zorder=slice_idx * 6 + 2, rasterized=True)
        ax.add_collection3d(pc)

    # ── Triangle border ────────────────────────────────────────────────────────
    n_pts = 100
    bx = np.concatenate([np.linspace(0, mv, n_pts), np.linspace(mv, 0, n_pts),
                          np.zeros(n_pts), [0]])
    by = np.concatenate([np.zeros(n_pts), np.linspace(0, mv, n_pts),
                          np.linspace(mv, 0, n_pts), [0]])
    bz = np.full_like(bx, z_disp)
    ax.plot(bx, by, bz, color=BORDER_COLOR, lw=BORDER_LW, alpha=0.95,
            zorder=slice_idx * 6 + 8)

    # ── Tick numbers ON slice — tight to triangle edges ───────────────────────
    if slice_idx > 0:
        tick_vals = [0, 20, 40, 60, 80, 100]
        for tv in tick_vals:
            if tv <= mv + 1e-6:
                # Along bottom edge (P_CHV = 0), label x-axis ticks
                ax.text(tv, -TICK_OFFSET, z_disp, f'{int(tv)}',
                        fontsize=FS_TICK_ON_SLICE, color=MID,
                        ha='center', va='top', zorder=slice_idx * 6 + 9)
                # Along left edge (P_AV = 0), label y-axis ticks
                ax.text(-TICK_OFFSET*1.5, tv, z_disp, f'{int(tv)}',
                        fontsize=FS_TICK_ON_SLICE, color=MID,
                        ha='right', va='center', zorder=slice_idx * 6 + 9)

    # ── White interior grid lines (clipped to triangle) ──────────────────────
    for gv in [20, 40, 60, 80]:
        if gv < mv - 1:
            ax.plot([gv, gv], [0, mv - gv], [z_disp, z_disp],
                    color='white', lw=GRID_LW, alpha=0.55, zorder=slice_idx * 6 + 4)
            ax.plot([0, mv - gv], [gv, gv], [z_disp, z_disp],
                    color='white', lw=GRID_LW, alpha=0.55, zorder=slice_idx * 6 + 4)

    # ── Slice label — placed just above the hypotenuse midpoint ──────────────
    # Hypotenuse midpoint: (mv/2, mv/2).  Shift slightly toward viewer.
    # lx = mv * 0.5
    # ly = mv * 0.5 + TICK_OFFSET * 1.5
    # ax.text(lx, ly, z_disp,
    #         f'$P_{{\\mathrm{{CAV}}}}={p_cav_pct:.0f}\\%$',
    #         fontsize=FS_SLICE_LABEL, fontweight='bold', color=DARK,
    #         ha='center', va='bottom', zorder=slice_idx * 6 + 10)
    lx = mv + TICK_OFFSET * 3 # 在最右端再往外推一点距离
    ly = TICK_OFFSET * 1.5                   # 对齐到 Y=0 的底边高度
    
    # 2. 修改对齐方式：水平靠左对齐 (ha='left')，垂直居中 (va='center')
    ax.text(lx, ly, z_disp,
            f'$P_{{\\mathrm{{CAV}}}}={p_cav_pct:.0f}\\%$',
            fontsize=FS_SLICE_LABEL, fontweight='bold', color=DARK,
            ha='right', va='center', zorder=slice_idx * 6 + 10)


# ─── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22), facecolor='white', dpi=300)
ax  = fig.add_axes([0.02, 0.02, 0.96, 0.88], projection='3d')
ax.set_facecolor("#FFFFFF")

for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#AAB7B8')
ax.grid(False)

# ─── Draw all slices ──────────────────────────────────────────────────────────
for slice_idx, p_cav in enumerate(SLICES):
    z_disp = Z_POSITIONS[slice_idx]
    print(f'  Rendering P_CAV={p_cav}%  z={z_disp}...')
    draw_slice(ax, p_cav, z_disp, slice_idx)

# ─── Vertical connector at origin corner ──────────────────────────────────────
ax.plot([0, 0], [0, 0], [0, Z_MAX],
        color='#7F8C8D', lw=1.0, ls=':', alpha=0.55, zorder=1)

# ─── Transition arrows ────────────────────────────────────────────────────────
# arrow_data = [
#     (0, 1, 'CL, NCAL emerge\nat $P_{\\mathrm{CAV}}=20\\%$', 50, 25),
#     (1, 2, 'CAL emerges\nat $P_{\\mathrm{CAV}}=40\\%$',     45, 18),
# ]
# for from_i, to_i, label, ta, tc in arrow_data:
#     z0 = Z_POSITIONS[from_i] + 80
#     z1 = Z_POSITIONS[to_i]   - 80
#     if z1 > z0:
#         ax.quiver(ta, tc, z0, 0, 0, z1 - z0,
#                   color='#1A3A5C', linewidth=2.0,
#                   arrow_length_ratio=0.12, alpha=0.9, zorder=90)
#         ax.text(ta + 3, tc, (z0 + z1) / 2, label,
#                 fontsize=FS_ARROW_LABEL, color='#1A3A5C', style='italic',
#                 ha='left', va='center', zorder=100)

# ─── Axes config ──────────────────────────────────────────────────────────────
PAD = TICK_OFFSET + 2
ax.set_xlim(-PAD, 105)
ax.set_ylim(-PAD, 105)
ax.set_zlim(0, Z_MAX + 100)

ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_zticks(Z_POSITIONS)
ax.set_zticklabels([f'{p}%' for p in SLICES])

ax.zaxis.set_tick_params(pad=20)
ax.xaxis.set_tick_params(pad=6)
ax.yaxis.set_tick_params(pad=6)

lkw = dict(fontsize=FS_AXIS_LABEL, fontweight='bold', color=DARK)
ax.set_xlabel(r'AV Penetration  $P_\mathrm{AV}$ / %',   labelpad=25, **lkw)
ax.set_ylabel(r'CHV Penetration  $P_\mathrm{CHV}$ / %', labelpad=25, **lkw)
ax.tick_params(axis='x', labelsize=FS_OUTER_TICKS, colors=MID, pad=6)
ax.tick_params(axis='y', labelsize=FS_OUTER_TICKS, colors=MID, pad=6)
ax.tick_params(axis='z', labelsize=FS_Z_TICKS,     colors=MID, pad=12)

# ── View angle: lower elevation = bigger apparent layer gap variation ─────────
ax.view_init(elev=17, azim=222)

# ─── Legend ───────────────────────────────────────────────────────────────────
legend_labels = {
    'GL':   'GL — General-purpose lanes',
    'AL':   'AL — AV-dedicated lanes',
    'CL':   'CL —  CV-dedicated lanes',
    'NCAL': 'NCAL — Non-CAV lanes',
    'CAL':  'CAL — CAV-dedicated lanes',
}
handles = [
    mpatches.Patch(facecolor=strategy_colors[s], edgecolor='#5D6D7E',
                   label=legend_labels[s], linewidth=0.9, alpha=0.93)
    for s in strategy_order_legend
]
leg = ax.legend(handles=handles, 
                loc='center left',           # 用图例框的“左边中点”去对齐
                bbox_to_anchor=(1.05, 0.5),  # 锚点放在图像右侧外部 (x=1.05) 和垂直居中处 (y=0.5)
                fontsize=FS_LEGEND, 
                framealpha=0.97, edgecolor='#BFC9CA', fancybox=False,
                frameon=True, handlelength=1.6, handleheight=1.3)

# ─── Titles ───────────────────────────────────────────────────────────────────
# fig.text(0.52, 0.975,
#          r'Optimal Strategy Transition Across $P_\mathrm{CAV}$ Levels — Layered Slice View',
#          ha='center', va='top', fontsize=FS_TITLE, fontweight='bold', color=DARK)
# fig.text(0.52, 0.945,
#          r'Each slice = one $P_\mathrm{CAV}$ level  $\vert$  '
#          r'Filled area = dominant optimal lane strategy  $\vert$  '
#          r'Shrinking triangular domain: $P_\mathrm{AV}+P_\mathrm{CHV} \leq 1-P_\mathrm{CAV}$  $\vert$  '
#          r'Arrows = critical transition thresholds',
#          ha='center', va='top', fontsize=FS_SUBTITLE, color='#566573', style='italic')

# ─── Save ─────────────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out = f'./plot1_strategy/fig_v3_{timestamp}.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}')