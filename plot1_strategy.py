import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Patch, Polygon
from matplotlib.colorbar import Colorbar
from matplotlib.colors import ListedColormap, Normalize
from scipy.interpolate import griddata
import sys, os

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAVE_DIR)

from optimizer import LaneOptimizer

matplotlib.rcParams['font.family'] = 'Palatino Linotype'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'

n = 5
L_max = 5

strategy_names = ['GL', 'NCAL', 'CL', 'AL', 'CAL']
strategy_colors_map = {
    'GL':   '#D0CECE',
    'NCAL': '#DEEBF7',
    'CL':   '#FFF2CC',
    'AL':   '#FBE5D6',
    'CAL':  '#E2F0D9',
}

automation_list = []
connectivity_list = []
strat_list = []
capacity_list = []

p_cav_list = []
p_chv_list = []
p_av_list = []
p_hv_list = []

step = 0.05
for p_cav in np.arange(0.0, 1.05, step):
    for p_chv in np.arange(0.0, 1.05 - p_cav, step):
        for p_av in np.arange(0.0, 1.05 - p_cav - p_chv, step):
            p_hv = 1.0 - p_cav - p_chv - p_av
            if p_hv < -0.01:
                continue
            p_hv = max(p_hv, 0.0)

            automation = p_cav + p_av
            connectivity = p_cav + p_chv

            opt = LaneOptimizer()
            result = opt.optimize(
                P_CAV=p_cav,
                P_CHV=p_chv,
                P_AV=p_av,
                P_HV=p_hv,
                n=n,
                L_max=L_max,
                verbose=False,
            )

            dominant = result['best_result']['strategy']
            total_capacity = result['best_result']['total_capacity']

            p_cav_list.append(p_cav)
            p_chv_list.append(p_chv)
            p_av_list.append(p_av)
            p_hv_list.append(p_hv)
            
            automation_list.append(automation)
            connectivity_list.append(connectivity)
            strat_list.append(dominant)
            capacity_list.append(total_capacity)

    print(f"  p_cav={p_cav:.2f} done")

automation_arr = np.array(automation_list) * 100
connectivity_arr = np.array(connectivity_list) * 100
strat_arr = np.array(strat_list)
capacity_arr = np.array(capacity_list)

data_table = np.column_stack([automation_arr, connectivity_list, strat_list, capacity_arr])
data_header = 'Automation,Connectivity,Strategy,Capacity'

with open(os.path.join(SAVE_DIR, 'strategy_data.csv'), 'w') as f:
    f.write(data_header + '\n')
    for i in range(len(automation_arr)):
        f.write(f'{automation_arr[i]:.2f},{connectivity_arr[i]:.2f},{strat_list[i]},{capacity_arr[i]:.2f}\n')
print("Data table saved to strategy_data.csv")

data_table2 = np.column_stack([p_cav_list, p_chv_list, p_av_list, p_hv_list, 
                              automation_list, connectivity_list, strat_list, capacity_list])
data_header2 = 'P_CAV,P_CHV,P_AV,P_HV,Automation,Connectivity,Strategy,Capacity'

with open(os.path.join(SAVE_DIR, 'penetration_strategy_data.csv'), 'w') as f:
    f.write(data_header2 + '\n')
    for i in range(len(automation_arr)):
        f.write(f'{p_cav_list[i]:.2f},{p_chv_list[i]:.2f},{p_av_list[i]:.2f},{p_hv_list[i]:.2f},{automation_list[i]:.2f},{connectivity_list[i]:.2f},{strat_list[i]},{capacity_arr[i]:.2f}\n')
print("Data table saved to strategy_data.csv")

x_grid = np.linspace(0, 100, 200)
y_grid = np.linspace(0, 100, 200)
X, Y = np.meshgrid(x_grid, y_grid)

Z = np.full_like(X, np.nan)
for i in range(len(automation_arr)):
    if 0 <= automation_arr[i] <= 100 and 0 <= connectivity_arr[i] <= 100:
        idx_x = np.argmin(np.abs(x_grid - automation_arr[i]))
        idx_y = np.argmin(np.abs(y_grid - connectivity_arr[i]))
        if np.isnan(Z[idx_y, idx_x]) or capacity_arr[i] > Z[idx_y, idx_x]:
            Z[idx_y, idx_x] = capacity_arr[i]

valid_mask = ~np.isnan(Z)
if np.any(valid_mask):
    points = np.array([X[valid_mask], Y[valid_mask]]).T
    values = Z[valid_mask]
    Z_filled = griddata(points, values, (X, Y), method='linear')
    Z_filled = np.nan_to_num(Z_filled, nan=np.nanmin(Z))
else:
    Z_filled = Z

fig, ax = plt.subplots(figsize=(10, 10))

half_step = 2.5
for i in range(len(automation_arr)):
    x, y = automation_arr[i], connectivity_arr[i]
    s = strat_arr[i]
    rect = Rectangle((x - half_step, y - half_step), step * 100, step * 100,
                     facecolor=strategy_colors_map[s],
                     edgecolor='white', linewidth=0.5)
    ax.add_patch(rect)

contour_levels = np.arange(10000, 27000, 2000)
cs = ax.contour(X, Y, Z_filled, levels=contour_levels, colors="#989696CE", 
                linewidths=1.5, linestyles='--', alpha=0.9)
labels = ax.clabel(cs, inline=True, fontsize=13, fmt='%d')
for label in labels:
    label.set_color('#333333')

ax.set_xlabel(r'Automation level ($P_{\mathrm{CAV}} + P_{\mathrm{AV}}$) / %', fontsize=20, fontweight='bold')
ax.set_ylabel(r'Connectivity level ($P_{\mathrm{CAV}} + P_{\mathrm{CHV}}$) / %', fontsize=20, fontweight='bold')

tick_positions = np.arange(0, 101, 5)
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.tick_params(axis='both', labelsize=14)

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
# ax.set_aspect('equal')
ax.set_aspect(0.9)
ax.grid(True, linestyle='--', alpha=0.3)

cmap_colors = [strategy_colors_map[s] for s in strategy_names]
cmap = ListedColormap(cmap_colors)
norm = Normalize(vmin=0, vmax=len(strategy_names))

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = Colorbar(cbar_ax, cmap=cmap, norm=norm, ticks=list(np.arange(0.5, len(strategy_names), 1)))
cbar.set_ticklabels(strategy_names)
cbar.set_label('Optimal lane strategy', fontsize=20)
cbar.ax.tick_params(labelsize=11)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(os.path.join(SAVE_DIR, 'fig_strategy_automation_connectivity_origin.png'),
            dpi=600, bbox_inches='tight')
plt.close()
print("Figure saved.")
