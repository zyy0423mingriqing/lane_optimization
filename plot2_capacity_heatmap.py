import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import griddata
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─── Typography ───────────────────────────────────────────────────────────────
matplotlib.rcParams['font.family']       = 'serif'
matplotlib.rcParams['font.serif']        = ['Palatino Linotype', 'Palatino',
                                             'Georgia', 'DejaVu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset']  = 'stix'

# ─── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv('penetration_capacity_data.csv')

# Take the maximum capacity at each (Automation, Connectivity) grid point
best = df.groupby(['Automation', 'Connectivity'])['Capacity'].max().reset_index()

# ─── Interpolation onto fine grid ─────────────────────────────────────────────
RES   = 400
x_lin = np.linspace(0, 100, RES)
y_lin = np.linspace(0, 100, RES)
X, Y  = np.meshgrid(x_lin, y_lin)

Z_raw = griddata(
    best[['Automation', 'Connectivity']].values,
    best['Capacity'].values,
    (X, Y),
    method='linear',
)
# Fill NaN corners with nearest
Z_near = griddata(
    best[['Automation', 'Connectivity']].values,
    best['Capacity'].values,
    (X, Y),
    method='nearest',
)
Z = np.where(np.isnan(Z_raw), Z_near, Z_raw)

# ─── Custom colormap: deep navy → indigo → rose → amber → cream ──────────────
# Designed to be perceptually uniform, print-safe, and visually refined
cmap_colors = [
    (0.06, 0.08, 0.24),   # deep navy
    (0.22, 0.12, 0.42),   # deep indigo
    (0.52, 0.12, 0.47),   # dark rose
    (0.82, 0.32, 0.22),   # burnt orange
    (0.97, 0.68, 0.18),   # amber
    (0.99, 0.96, 0.82),   # warm cream
]
cmap = LinearSegmentedColormap.from_list('refined_plasma', cmap_colors, N=512)

# ─── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 8), facecolor='white')

# ── Heatmap
hm = ax.pcolormesh(X, Y, Z,
                   cmap='viridis',
                   shading='gouraud',   # smooth bilinear shading
                   vmin=9000, vmax=31500,
                   rasterized=True,
                   zorder=1)

# ─── Contour lines ────────────────────────────────────────────────────────────
# Major contours (labelled) — only 6 selected to avoid crowding
levels_major = [10000, 12000, 14000, 16000, 18000, 20000, 24000, 28000]
cs_major = ax.contour(X, Y, Z,
                      levels=levels_major,
                      colors='white',
                      linewidths=1.2,
                      linestyles='-',
                      alpha=0.90,
                      zorder=3)


def fmt_k(val, _=None):
    # return f'{int(val/1000):d} ($\\times 10^3$)'
    return f'{int(val):d} '

clabels = ax.clabel(cs_major,
                    inline=True,
                    fontsize=12,
                    fmt=fmt_k,
                    inline_spacing=6,
                    use_clabeltext=True,
                    manual=False)
for lbl in clabels:
    lbl.set_color('white')
    lbl.set_fontweight('bold')
    lbl.set_alpha(0.95)
    lbl.set_backgroundcolor((0, 0, 0, 0))

# ─── P_CAV direction arrow ────────────────────────────────────────────────────
arrow_start = (5,  5)
arrow_end   = (92, 92)

ax.annotate('', xy=(100, 100), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color="#FDFFFD", lw=3, 
                           mutation_scale=20))

# Label placed beside the arrow, slightly offset perpendicular to it
ax.text(
    60, 73,
    r'Increasing $P_{\mathrm{CAV}}$',
    fontsize=12, fontweight='bold',
    color="#040404",
    ha='center', va='center',
    rotation=43,
    bbox=dict(boxstyle='round,pad=0.30',
              facecolor="#FAFAFA", edgecolor="#F8FAF9",
              linewidth=1.2, alpha=0.84),
    zorder=9,
)

# Dashed diagonal reference line (very subtle)
diag = np.linspace(0, 100, 200)
ax.plot(diag, diag,
        color='white', lw=0.5, ls='--', alpha=0.20, zorder=2)

# ─── Colorbar ─────────────────────────────────────────────────────────────────
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='4%', pad=0.12)

cbar = fig.colorbar(hm, cax=cax)
cbar.set_label(r'Total Capacity (veh/h)',
               fontsize=12, fontweight='bold',
               color='#1C2833', labelpad=10)
cbar.ax.tick_params(labelsize=12, colors='#1C2833', length=3)
cbar.outline.set_edgecolor('#AAAAAA')
cbar.outline.set_linewidth(0.8)

# Custom ticks at round values
cbar_ticks = [9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'{v//1000:d},000' for v in cbar_ticks])

# ─── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_aspect('equal')

ax.set_xticks(range(0, 101, 10))
ax.set_yticks(range(0, 101, 10))
ax.tick_params(axis='both', labelsize=12, colors='#1C2833',
               length=4, width=0.8, direction='out')

ax.set_xlabel(r'Automation Level  $(P_{\mathrm{CAV}} + P_{\mathrm{AV}})$ / %',
              fontsize=13, fontweight='bold', color='#1C2833', labelpad=8)
ax.set_ylabel(r'Connectivity Level  $(P_{\mathrm{CAV}} + P_{\mathrm{CHV}})$ / %',
              fontsize=13, fontweight='bold', color='#1C2833', labelpad=8)

# Subtle grid
ax.grid(True, linestyle=':', linewidth=0.45, alpha=0.30,
        color='white', zorder=0)

# Clean spines
for spine in ax.spines.values():
    spine.set_edgecolor('#888888')
    spine.set_linewidth(0.8)

# ─── Layout & save ────────────────────────────────────────────────────────────
fig.tight_layout()

out = 'fig_capacity_heatmap.png'
plt.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}')