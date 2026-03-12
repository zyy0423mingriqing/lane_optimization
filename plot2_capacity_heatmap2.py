import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ─── Typography ───────────────────────────────────────────────────────────────
rcParams['font.family']          = 'serif'
rcParams['font.serif']           = ['Palatino Linotype', 'Palatino',
                                     'Georgia', 'DejaVu Serif']
rcParams['axes.unicode_minus']   = False
rcParams['mathtext.fontset']     = 'stix'

# ─── 1. Data ──────────────────────────────────────────────────────────────────
df   = pd.read_csv('./plot2_capacity_heatmap/capacity_heatmap_data.csv')
best = df.groupby(['Automation', 'Connectivity'])['Capacity'].max().reset_index()
best.columns = ['Automation', 'Connectivity', 'Capacity']

x_raw = best['Automation'].values
y_raw = best['Connectivity'].values
z_raw = best['Capacity'].values

# ─── 2. Interpolation onto fine grid ─────────────────────────────────────────
RES   = 200
xi    = np.linspace(x_raw.min(), x_raw.max(), RES)
yi    = np.linspace(y_raw.min(), y_raw.max(), RES)
Xi, Yi = np.meshgrid(xi, yi)

Zi_raw = griddata((x_raw, y_raw), z_raw, (Xi, Yi), method='cubic')
Zi_nst = griddata((x_raw, y_raw), z_raw, (Xi, Yi), method='nearest')
Zi     = np.where(np.isnan(Zi_raw), Zi_nst, Zi_raw)

# ─── 3. Custom colormap  (navy → indigo → rose → amber → cream) ───────────────
# cmap_colors = [
#     (0.06, 0.08, 0.24),   # deep navy
#     (0.22, 0.12, 0.42),   # deep indigo
#     (0.52, 0.12, 0.47),   # dark rose
#     (0.82, 0.32, 0.22),   # burnt orange
#     (0.97, 0.68, 0.18),   # amber
#     (0.99, 0.96, 0.82),   # warm cream
# ]
# cmap_surface = LinearSegmentedColormap.from_list('refined_plasma', cmap_colors, N=512)

# ─── 4. Figure & 3D axes ──────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9), facecolor='white')
ax  = fig.add_subplot(111, projection='3d', facecolor='white')

# ─── 4a. 3D surface ───────────────────────────────────────────────────────────
surf = ax.plot_surface(
    Xi, Yi, Zi,
    cmap='terrain',
    linewidth=0,
    antialiased=True,
    alpha=0.93,
    rcount=RES,
    ccount=RES,
    vmin=z_raw.min(),
    vmax=z_raw.max(),
)

# ─── 4b. Bottom projection plane ─────────────────────────────────────────────
# z_offset = z_raw.min() - (z_raw.max() - z_raw.min()) * 0.42
z_offset = -50000

# Filled contour projection
ax.contourf(
    Xi, Yi, Zi,
    levels=20,
    cmap='terrain',
    alpha=0.30,
    offset=z_offset,
    zdir='z',
    vmin=z_raw.min(),
    vmax=z_raw.max(),
)

# Labelled contour lines — only major levels to avoid crowding
levels_major = np.linspace(z_raw.min(), z_raw.max(), 9).astype(int)
cs = ax.contour(
    Xi, Yi, Zi,
    levels=levels_major,
    cmap='gray',
    linewidths=0.7,
    alpha=0.75,
    offset=z_offset,
    zdir='z',
)

# ─── 4c. P_CAV direction indicator on bottom plane ───────────────────────────
z_plane = z_offset

# Subtle dashed diagonal reference line
diag = np.linspace(0, 100, 200)
ax.plot(diag, diag,
        zs=z_plane, zdir='z',
        color='white', lw=0.7, ls='--', alpha=0.35, zorder=2)

# Arrow via quiver (stays in z = z_plane)
ax.quiver(
    5, 5, z_plane,
    87, 87, 0,
    length=1.0, normalize=False,
    color='#FDFFFD',
    linewidth=2.2,
    arrow_length_ratio=0.07,
    zorder=10,
)

# Text label
ax.text(
    58, 71, z_plane + (z_raw.max() - z_raw.min()) * 0.02,
    r'Increasing $P_{\mathrm{CAV}}$',
    fontsize=11, fontweight='bold',
    color='#040404',
    ha='center', va='center',
    zdir=None,
    bbox=dict(
        boxstyle='round,pad=0.30',
        facecolor='#FAFAFA', edgecolor='#CCCCCC',
        linewidth=1.0, alpha=0.85,
    ),
    zorder=11,
)

# ─── 4d. Axis labels ──────────────────────────────────────────────────────────
ax.set_xlabel(
    r'Automation Level  $(P_{\mathrm{CAV}}+P_{\mathrm{AV}})$ / %',
    fontsize=11, fontweight='bold', color='#1C2833', labelpad=14,
)
ax.set_ylabel(
    r'Connectivity Level  $(P_{\mathrm{CAV}}+P_{\mathrm{CHV}})$ / %',
    fontsize=11, fontweight='bold', color='#1C2833', labelpad=14,
)
# ax.set_zlabel(
#     r'Total Capacity (veh/h)',
#     fontsize=11, fontweight='bold', color='#1C2833', labelpad=12,
# )

# Tick style
ax.tick_params(axis='both', labelsize=9, colors='#1C2833',
               length=3, width=0.7)
ax.set_xticks(range(0, 101, 20))
ax.set_yticks(range(0, 101, 20))

# ─── 4e. View angle ───────────────────────────────────────────────────────────
ax.view_init(elev=28, azim=-50)

# ─── 4f. Z-axis range (extend down to projection plane) ───────────────────────
ax.set_zlim(z_offset, z_raw.max() * 1.1)

# Custom Z tick labels formatted as "xx,000"
z_ticks = np.linspace(
    int(np.ceil(z_raw.min() / 1000)-1) * 1000,
    int(np.floor(z_raw.max() / 1000)+1) * 1000,
    4,
).astype(int)
ax.set_zticks(z_ticks)
ax.set_zticklabels([f'{v:,}' for v in z_ticks], fontsize=8)

# ─── 4g. Colorbar ─────────────────────────────────────────────────────────────
cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.52,
    aspect=16,
    pad=0.06,
)
cbar.set_label(r'Total Capacity (veh/h)',
               fontsize=11, fontweight='bold',
               color='#1C2833', labelpad=10)
cbar.ax.tick_params(labelsize=9, colors='#1C2833', length=3)
cbar.outline.set_edgecolor('#AAAAAA')
cbar.outline.set_linewidth(0.8)

# Round tick labels for colorbar
cb_ticks = np.linspace(z_raw.min(), z_raw.max(), 7).astype(int)
cbar.set_ticks(cb_ticks)
cbar.set_ticklabels([f'{v:,}' for v in cb_ticks])

# ─── 4h. Pane & grid styling (matches code-2 aesthetic) ──────────────────────
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#CCCCCC')
ax.yaxis.pane.set_edgecolor('#CCCCCC')
ax.zaxis.pane.set_edgecolor('#CCCCCC')
ax.grid(True, linestyle=':', linewidth=0.4, color='grey', alpha=0.40)

# ─── 5. Save ──────────────────────────────────────────────────────────────────
plt.tight_layout()
out = 'fig_capacity_heatmap_3d.png'
plt.savefig(out, dpi=1000, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved → {out}')