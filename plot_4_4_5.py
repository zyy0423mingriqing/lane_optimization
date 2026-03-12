import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import sys, os
import pandas as pd

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SAVE_DIR)

from optimizer import LaneOptimizer
from config import VEHICLE_TYPES

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.fontset'] = 'stix'

L_max = 8

STRATEGY_COLORS = {
    'GL':   '#D0CECE',
    'NCAL': '#DEEBF7',
    'CL':   '#FFF2CC',
    'AL':   '#FBE5D6',
    'CAL':  '#E2F0D9',
}

STACK_ORDER = ['GL', 'NCAL', 'CL', 'AL', 'CAL']

sweep_range = np.linspace(0.0, 1.0, 20)

for N_lanes in [3, 4, 5]:
    cap_by_strategy = {s: [] for s in STACK_ORDER}

    for p_cav in sweep_range:
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
            n=N_lanes,
            L_max=L_max,
            verbose=False,
        )

        strat_cap = {s: 0.0 for s in STACK_ORDER}
        lane_allocation = result['best_result']['lane_allocation']
        for lane_s, lane_c in lane_allocation.items():
            strat_cap[lane_s] += lane_c

        for s in STACK_ORDER:
            cap_by_strategy[s].append(strat_cap[s])

    print(f"  N_lanes={N_lanes} sweep done")

    # Convert to arrays
    x = sweep_range * 100
    stacks = {s: np.array(cap_by_strategy[s]) for s in STACK_ORDER}

    # Export data to CSV
    df_data = {'CAV_Penetration': x.copy()}
    for s in STACK_ORDER:
        df_data[s] = stacks[s].copy()
    df_data['Total_Capacity'] = sum(stacks[s] for s in STACK_ORDER)
    df = pd.DataFrame(df_data)
    df = df[['CAV_Penetration', 'GL', 'NCAL', 'CL', 'AL', 'CAL', 'Total_Capacity']]
    csv_path = os.path.join(SAVE_DIR, f'fig_4_4_5_capacity_{N_lanes}lanes.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data exported to {csv_path}")

    # Build stacked area chart
    fig, ax = plt.subplots(figsize=(10, 6))

    y_bottom = np.zeros_like(x)
    for s in STACK_ORDER:
        y_vals = stacks[s]
        if np.max(y_vals) < 1e-3:
            continue
        ax.fill_between(x, y_bottom, y_bottom + y_vals,
                        color=STRATEGY_COLORS[s], alpha=0.85,
                        edgecolor='white', linewidth=0.5)
        # Add text annotation in the middle of the region where it's large enough
        mask = y_vals > 50
        if np.any(mask):
            indices = np.where(mask)[0]
            mid_idx = indices[len(indices) // 2]
            mid_x = x[mid_idx]
            mid_y = y_bottom[mid_idx] + y_vals[mid_idx] / 2
            if y_vals[mid_idx] > 200:
                ax.text(mid_x, mid_y, s, fontsize=10, fontweight='bold',
                        ha='center', va='center', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=STRATEGY_COLORS[s],
                                  edgecolor='none', alpha=0.7))
        y_bottom = y_bottom + y_vals

    # Plot total capacity line on top
    total_cap = sum(stacks[s] for s in STACK_ORDER)
    ax.plot(x, total_cap, 'k-', linewidth=1.5, alpha=0.6)

    # Legend
    legend_elements = [Patch(facecolor=STRATEGY_COLORS[s], edgecolor='white',
                             alpha=0.85, label=s)
                       for s in STACK_ORDER if np.max(stacks[s]) > 1e-3]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper left',
              framealpha=0.95, edgecolor='black', fancybox=False)

    ax.set_xlabel('CAV Penetration Rate / %', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Segment Capacity / (veh/h)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.35, color='grey')
    ax.set_xlim([0, 95])
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fname = f'.\\plot1_strategy\\fig_4_4_5_capacity_breakdown_{N_lanes}lanes.png'
    plt.savefig(os.path.join(SAVE_DIR, fname), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Fig 4-4-5 ({N_lanes} lanes) saved.")

print("All Fig 4-4-5 sub-figures saved.")
