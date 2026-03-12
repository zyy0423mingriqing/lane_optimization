"""
绘制 CAV 渗透率 vs 吞吐量的分组堆叠柱状图
--------------------------------------------------------------
数据来源：调用 LaneOptimizer 获取最优策略的车道容量
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

from optimizer import LaneOptimizer
from config import VEHICLE_TYPES

matplotlib.rcParams['font.family'] = 'Palatino Linotype'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'

STRATEGY_COLORS = {
    'GL':   '#D0CECE',
    'NCAL': '#DEEBF7',
    'CL':   '#FFF2CC',
    'AL':   '#FBE5D6',
    'CAL':  '#E2F0D9',
}

LABEL_COLOR = {
    'GL':   '#555555',
    'NCAL': '#1A5276',
    'CL':   '#7D6608',
    'AL':   '#922B21',
    'CAL':  '#0B5345',
}

L_max = 5

STACK_ORDER = ['GL', 'NCAL', 'CL', 'AL', 'CAL']

PENETRATION_RATES = np.round(np.arange(0.0, 1.1, 0.1), 1)


def get_optimal_result(p_cav: float, n_lanes: int, L_max: int) -> dict:
    """
    调用优化器获取最优策略结果
    """
    remain = 1.0 - p_cav
    others = [v for v in VEHICLE_TYPES if v != 'CAV']
    P_seg = {v: remain / 3.0 for v in others}
    P_seg['CAV'] = p_cav

    opt = LaneOptimizer()
    result = opt.optimize(
        P_CAV=P_seg['CAV'],
        P_CHV=P_seg['CHV'],
        P_AV=P_seg['AV'],
        P_HV=P_seg['HV'],
        n=n_lanes,
        L_max=L_max,
        verbose=False,
    )

    lane_allocation = result['best_result']['lane_allocation']
    lane_capacities = result['best_result']['lane_capacities']
    total_capacity = result['best_result']['total_capacity']

    return {
        'strategy': result['best_result']['strategy'],
        'lane_allocation': lane_allocation,
        'lane_capacities': lane_capacities,
        'total_capacity': round(total_capacity, 2),
    }


def collect_data(n_lanes: int, L_max: int) -> list:
    """
    收集指定车道数下所有渗透率的最优结果
    """
    results = []
    for p in PENETRATION_RATES:
        f1 = get_optimal_result(p, n_lanes, L_max)
        results.append({'p': p, 'f1': f1})
    return results


def plot_throughput(data: list,
                    n_lanes: int,
                    y_max: int = 12000,
                    y_step: int = 2000,
                    bar_width: float = 0.25,
                    gap_within_group: float = 0.01,
                    figsize: tuple = (13, 6),
                    save_path: str = 'fig_cav_throughput.png',
                    dpi: int = 1000):
    """
    绘制分组堆叠柱状图
    """
    n = len(data)
    x = np.arange(n)
    f_keys = ['f1']
    total_w = len(f_keys) * bar_width + (len(f_keys) - 1) * gap_within_group
    offsets = [
        -total_w / 2 + bar_width / 2 + i * (bar_width + gap_within_group)
        for i in range(len(f_keys))
    ]

    fig, ax = plt.subplots(figsize=figsize)

    for fi, (fkey, offset) in enumerate(zip(f_keys, offsets)):
        bottoms = np.zeros(n)

        for s in STACK_ORDER:
            heights = np.array([
                row[fkey]['lane_capacities'].get(s, 0.0) * row[fkey]['lane_allocation'].get(s, 0)
                for row in data
            ])

            bars = ax.bar(
                x + offset,
                heights,
                bottom=bottoms,
                width=bar_width,
                color=STRATEGY_COLORS[s],
                edgecolor='#888888',
                linewidth=0.4,
                zorder=2,
            )

            for bar_i, (bar, h, bot) in enumerate(zip(bars, heights, bottoms)):
                if h < 200:
                    continue
                mid_y = bot + h / 2
                lane_count = data[bar_i][fkey]['lane_allocation'].get(s, 0)
                label = f"{s}\n({lane_count})" if lane_count > 0 else s
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mid_y,
                    label,
                    ha='center', va='center',
                    fontsize=7.5,
                    color=LABEL_COLOR[s],
                    fontweight='normal',
                    clip_on=True,
                )

            bottoms += heights

    legend_colors = ['#5DCAA5']
    legend_labels = [r'$f_1$ (Optimal)']
    patches = [
        mpatches.Patch(facecolor=c, edgecolor='#888888', linewidth=0.5, label=l)
        for c, l in zip(legend_colors, legend_labels)
    ]
    ax.legend(
        handles=patches,
        loc='upper left',
        frameon=True,
        framealpha=0.9,
        edgecolor='#cccccc',
        fontsize=11,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f'{row["p"]:.1f}' for row in data], fontsize=11)
    ax.set_xlim(-0.6, n - 0.4)

    # ax.set_ylim(0, y_max)
    # ax.set_yticks(range(0, y_max + 1, y_step))
    ax.set_yticklabels([f'{v:,}' for v in range(0, y_max + 1, y_step)], fontsize=11)
    ax.set_ylabel('Total Capacity (veh/h)', fontsize=12)
    ax.set_xlabel('CAV Penetration\n$P_{CAV}$', fontsize=12)
    # ax.set_title(f'Throughput vs CAV Penetration Rate (N={n_lanes} lanes, L_max={L_max})', fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f'图像已保存至: {save_path}')
    plt.close()


if __name__ == '__main__':

    y_max_values = {3: 9000, 4: 12000, 5: 15000}
    y_step_values = {3: 1500, 4: 2000, 5: 2500}

    for n_lanes in [3, 4, 5]:
        print(f"\n=== Processing {n_lanes} lanes ===")
        data = collect_data(n_lanes, L_max)

        y_max = y_max_values.get(n_lanes, 12000)
        
        y_step = y_step_values.get(n_lanes, 2000)

        plot_throughput(
            data,
            n_lanes=n_lanes,
            y_max=y_max,
            y_step=y_step,
            bar_width=0.6,
            gap_within_group=0.02,
            figsize=(13, 6),
            save_path=f'fig_cav_throughput_n{n_lanes}.png',
            dpi=1000,
        )

    print("\n=== All figures generated successfully ===")
