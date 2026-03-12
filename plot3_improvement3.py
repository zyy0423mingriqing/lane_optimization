import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib import rcParams

# ─── Typography ───────────────────────────────────────────────────────────────
rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Palatino Linotype', 'Palatino', 'Georgia', 'DejaVu Serif']
rcParams['axes.unicode_minus'] = False
rcParams['mathtext.fontset'] = 'stix'

# ── 1. 读取数据 ──────────────────────────────────────────────────────────────
df = pd.read_csv("./plot3_penertation_improvement/improvement_data_max.csv")   # 修改为你的实际路径

x_raw = df["Automation"].values       # X 轴：自动化水平 (P_CAV + P_AV) / %
y_raw = df["Connectivity"].values     # Y 轴：联网水平  (P_CAV + P_CHV) / %
z_raw = df["Improvement_Pct"].values  # Z 轴：相对 GL 策略的提升百分比

# ── 2. 插值到均匀网格 ────────────────────────────────────────────────────────
xi = np.linspace(x_raw.min(), x_raw.max(), 120)
yi = np.linspace(y_raw.min(), y_raw.max(), 120)
Xi, Yi = np.meshgrid(xi, yi)

Zi = griddata(
    points=(x_raw, y_raw),
    values=z_raw,
    xi=(Xi, Yi),
    method="cubic",
)

# 边缘 NaN 用 nearest 填充，避免空洞
Zi_nearest = griddata(
    points=(x_raw, y_raw),
    values=z_raw,
    xi=(Xi, Yi),
    method="nearest",
)
mask = np.isnan(Zi)
Zi[mask] = Zi_nearest[mask]

# ── 3. 绘图 ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection="3d")

# ── 3a. 3D 曲面 ──────────────────────────────────────────────────────────────
surf = ax.plot_surface(
    Xi, Yi, Zi,
    cmap='terrain',
    linewidth=0,
    antialiased=True,
    alpha=0.92,
    rcount=120,
    ccount=120,
)

# ── 3b. 底部等高线投影 ───────────────────────────────────────────────────────
z_offset = -100

contour = ax.contour(
    Xi, Yi, Zi,
    levels=20,
    cmap='terrain',
    linewidths=0.9,
    offset=z_offset,
    zdir="z",
)

contourf = ax.contourf(
    Xi, Yi, Zi,
    levels=20,
    cmap="RdYlGn",
    alpha=0.25,
    offset=z_offset,
    zdir="z",
)

# ── 3c. 底部投影平面上的指示线（3D 版）──────────────────────────────────────
z_plane = z_offset  # 固定在投影平面

# --- 对角参考虚线（很淡，z = z_plane）---
diag = np.linspace(0, 100, 200)
ax.plot(diag, diag,
        zs=z_plane, zdir='z',
        color='white', lw=0.8, ls='--', alpha=0.35, zorder=2)

# --- 主箭头：用 quiver 在 3D 投影平面上绘制 ---
#   quiver(x, y, z,  u, v, w)
# arrow_x0, arrow_y0 = 5,  5
# arrow_dx, arrow_dy = 87, 87   # 终点约 (92, 92)

# ax.quiver(
#     arrow_x0, arrow_y0, z_plane+1,   # 起点
#     arrow_dx, arrow_dy, 0,          # 方向向量（z=0 保持在平面内）
#     length=1.0,
#     normalize=True,
#     color="#000000",
#     linewidth=2.5,
#     arrow_length_ratio=0.06,        # 箭头头部比例
#     zorder=16,
# )

# --- 文字标注（3D ax.text，固定 z = z_plane）---
ax.text(
    60, 73, z_plane + 1,            # +1 稍微浮出平面，避免被遮住
    r'Increasing $P_{\mathrm{CAV}}$',
    fontsize=13, fontweight='bold',
    color="#040404",
    ha='center', va='center',
    zdir=None,                       # 文字始终朝向屏幕
    bbox=dict(
        boxstyle='round,pad=0.30',
        facecolor="#FAFAFA",
        edgecolor="#F8FAF9",
        linewidth=1.2,
        alpha=0.84,
    ),
    zorder=11,
)

# ── 3d. 坐标轴标签 ───────────────────────────────────────────────────────────
ax.set_xlabel(
    r'Automation Level  $(P_{\mathrm{CAV}}+P_{\mathrm{AV}})$ / %',
    fontsize=12, fontweight='bold', color='#1C2833', labelpad=10,
)
ax.set_ylabel(
    r'Connectivity Level  $(P_{\mathrm{CAV}}+P_{\mathrm{CHV}})$ / %',
    fontsize=12, fontweight='bold', color='#1C2833', labelpad=10,
)
# ax.set_zlabel(
#     "Improvement over GL / %",
#     fontsize=13, labelpad=10
# )
ax.tick_params(axis='both', labelsize=13, colors='#1C2833',
               length=3, width=0.7)

# ── 3e. 视角 ─────────────────────────────────────────────────────────────────
ax.view_init(elev=28, azim=-50)

# ── 3f. Z 轴范围 ─────────────────────────────────────────────────────────────
ax.set_zlim(-100, 40)

# Custom Z tick labels formatted as "xx,000"
z_ticks = np.linspace(
    int(np.ceil(z_raw.min())),
    int(np.floor(z_raw.max())+5),
    5,
).astype(int)
ax.set_zticks(z_ticks)
ax.set_zticklabels([f'{v:,}' for v in z_ticks], fontsize=13)

# ── 3g. 颜色条 ───────────────────────────────────────────────────────────────
cbar = fig.colorbar(
    surf,
    ax=ax,
    shrink=0.55,
    aspect=14,
    pad=0.08,
    label="Improvement over GL / %",
)
cbar.set_label(r'Improvement over GL / %',
               fontsize=13, fontweight='bold',
               color='#1C2833', labelpad=10)
cbar.ax.tick_params(labelsize=13)

# ── 3h. 背景美化 ─────────────────────────────────────────────────────────────
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("lightgrey")
ax.yaxis.pane.set_edgecolor("lightgrey")
ax.zaxis.pane.set_edgecolor("lightgrey")
ax.grid(True, linestyle="--", linewidth=0.4, color="grey", alpha=0.5)

plt.tight_layout()
plt.savefig("surface_contour_output_max.png", dpi=1000, bbox_inches="tight")
print("图像已保存为 surface_contour_output_max.png")
plt.show()