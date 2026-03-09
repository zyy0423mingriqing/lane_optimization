"""
Controlled experiments for Baseline (ALL GL) vs Optimized (Elastic Dedicated Lane).

Requirements implemented:
1) Directly imports and calls existing optimization model from main.py (LaneOptimizer).
2) Four experiments with x-axis in [0%, 100%], step 5%.
3) Other three vehicle penetrations are equally split from the remainder.
4) High-quality line plots with Palatino Linotype font.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from datetime import datetime
import csv

import matplotlib.pyplot as plt
import numpy as np

from optimizer import LaneOptimizer
from capacity_calculator import LaneCapacityCalculator


VEHICLES = ["CAV", "CHV", "AV", "HV"]
EXPERIMENT_ALLOWED_STRATEGIES = {
    "CAV": ["CAL","CL","AL","GL"],
    "CHV": ["CL","NCAL","GL"],
    "AV": ["AL","NCAL","GL"],
    "HV": ["NCAL","GL"],
}


def build_penetration_scenarios(target_vehicle: str, step: float = 0.05) -> List[Dict[str, float]]:
    """
    Build scenarios where target vehicle penetration varies from 0 to 1,
    and the other three types share the remainder equally.
    """
    scenarios: List[Dict[str, float]] = []
    points = int(round(1.0 / step)) + 1

    for i in range(points):
        p_target = round(i * step, 10)
        p_other = (1.0 - p_target) / 3.0

        p = {v: p_other for v in VEHICLES}
        p[target_vehicle] = p_target

        total = sum(p.values())
        p["HV"] += 1.0 - total
        scenarios.append(p)

    return scenarios


def compute_baseline_capacity(
    p_segment: Dict[str, float],
    n_lanes: int,
    l_max: int,
) -> float:
    """
    Baseline: ALL GL with uniform mixture on each lane.
    Under this assumption each GL lane has identical composition equal to segment penetration.
    """
    calculator = LaneCapacityCalculator()
    lane_capacity = calculator.calculate_capacity("GL", p_segment, l_max)
    return n_lanes * lane_capacity


def compute_optimized_capacity(
    p_segment: Dict[str, float],
    n_lanes: int,
    l_max: int,
    verbose: bool,
    allowed_strategies: List[str] | None = None,
    n_runs: int = 1,
) -> float:
    """
    Optimized group: directly call existing optimizer.
    When n_runs > 1, run the optimizer multiple times and return the average total capacity.
    """
    optimizer = LaneOptimizer()

    capacities: List[float] = []
    for _ in range(n_runs):
        result = optimizer.optimize(
            P_CAV=p_segment["CAV"],
            P_CHV=p_segment["CHV"],
            P_AV=p_segment["AV"],
            P_HV=p_segment["HV"],
            n=n_lanes,
            L_max=l_max,
            verbose=verbose,
        )
        capacities.append(result["best_result"]["total_capacity"])

    return float(np.mean(capacities))


def run_single_experiment(
    target_vehicle: str,
    n_lanes: int = 5,
    l_max: int = 5,
    step: float = 0.05,
    verbose: bool = False,
    n_runs: int = 1,
) -> Dict[str, List[float]]:
    """Run one penetration-rate sweep experiment."""
    scenarios = build_penetration_scenarios(target_vehicle=target_vehicle, step=step)

    x_percent: List[float] = []
    baseline_cap: List[float] = []
    optimized_cap: List[float] = []

    for p in scenarios:
        x_percent.append(p[target_vehicle] * 100)
        baseline_cap.append(compute_baseline_capacity(p, n_lanes=n_lanes, l_max=l_max))
        optimized_cap.append(
            compute_optimized_capacity(
                p,
                n_lanes=n_lanes,
                l_max=l_max,
                verbose=verbose,
                allowed_strategies=EXPERIMENT_ALLOWED_STRATEGIES.get(target_vehicle),
                n_runs=n_runs,
            )
        )

    improvement_pct = [
        (opt - base) / base * 100 if abs(base) > 1e-9 else 0.0
        for base, opt in zip(baseline_cap, optimized_cap)
    ]

    return {
        "x_percent": x_percent,
        "baseline": baseline_cap,
        "optimized": optimized_cap,
        "improvement_pct": improvement_pct,
    }


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "Palatino Linotype"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300


def save_experiment_results_table(
    target_vehicle: str,
    result: Dict[str, List[float]],
    output_dir: Path,
) -> None:
    """Save the experiment results used for plotting to a CSV table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"comparison_{target_vehicle.lower()}_data.csv"
    scenarios = build_penetration_scenarios(target_vehicle=target_vehicle, step=0.05)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "target_vehicle",
                "penetration_percent",
                "P_CAV",
                "P_CHV",
                "P_AV",
                "P_HV",
                "baseline_capacity",
                "optimized_capacity",
                "improvement_pct",
            ]
        )

        for scenario, x, baseline, optimized, improvement in zip(
            scenarios,
            result["x_percent"],
            result["baseline"],
            result["optimized"],
            result["improvement_pct"],
        ):
            writer.writerow(
                [
                    target_vehicle,
                    x,
                    scenario["CAV"],
                    scenario["CHV"],
                    scenario["AV"],
                    scenario["HV"],
                    baseline,
                    optimized,
                    improvement,
                ]
            )


def plot_comparison(
    target_vehicle: str,
    result: Dict[str, List[float]],
    output_dir: Path,
) -> None:
    """Create one high-quality comparison plot with a right-side improvement axis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    x = np.asarray(result["x_percent"], dtype=float)
    baseline = np.asarray(result["baseline"], dtype=float) / 1000
    optimized = np.asarray(result["optimized"], dtype=float) / 1000
    improvement = np.asarray(result["improvement_pct"], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax2 = ax.twinx()

    bar_width = 3.4
    improvement_bars = ax2.bar(
        x,
        improvement,
        width=bar_width,
        color="#b8d3a8",
        alpha=0.72,
        label="Improvement Rate",
        zorder=0,
    )

    ax.fill_between(x, baseline, optimized, color="#d89a9a", alpha=0.18, zorder=1)

    baseline_line, = ax.plot(
        x,
        baseline,
        color="#3f6fc0",
        marker="s",
        markersize=7.8,
        markerfacecolor="#4b7db7",
        markeredgecolor="#4b7db7",
        markeredgewidth=1.0,
        linewidth=2.4,
        label="No Dedicated Lane (All GL)",
        zorder=1,
    )

    optimized_line, = ax.plot(
        x,
        optimized,
        color="#c6534c",
        marker="o",
        markersize=8.0,
        markerfacecolor="#c6534c",
        markeredgecolor="#c6534c",
        markeredgewidth=1.0,
        linewidth=2.4,
        label="Elastic Dedicated Lane (Proposed)",
        zorder=2,
    )

    ax.set_xlabel(f"{target_vehicle} Penetration Rate / %", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"Total Segment Capacity / ($\times 10^3$ veh/h)", fontsize=18, fontweight="bold")
    ax2.set_ylabel("Capacity Improvement / %", fontsize=18, fontweight="bold", color="#6ea64b")

    ax.set_xlim(x.min(), x.max())
    ax.set_xticks(x)

    # if target_vehicle in ["AV", "CHV", "HV"]:
    #     ax.set_ylim(9, 13)
    #     ax2.set_ylim(0, 12)
    # else:
    #     y_min = min(baseline.min(), optimized.min())
    #     y_max = max(baseline.max(), optimized.max())
    #     ax.set_ylim(y_min, y_max)
    #     # ax.set_ylim(10, 27)
    #     ax2.set_ylim(0, 14)

    ax.tick_params(axis="both", labelsize=14, width=1.0, length=5)
    ax2.tick_params(axis="y", labelsize=14, colors="#6ea64b", width=1.0, length=5)

    ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.35)
    ax.set_axisbelow(True)

    legend_left = ax.legend(
        handles=[baseline_line, optimized_line],
        loc="upper left",
        fontsize=14,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        borderpad=0.5,
        handlelength=2.4,
    )
    ax.add_artist(legend_left)

    ax2.legend(
        handles=[improvement_bars],
        loc="upper right",
        fontsize=14,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        borderpad=0.5,
    )

    fig.tight_layout()
    fig.savefig(output_dir / f"comparison_{target_vehicle.lower()}.png", bbox_inches="tight")
    plt.close(fig)


def run_all_experiments(
    output_dir: str = "experiment_figures",
    verbose_solver: bool = False,
    n_runs: int = 1,
) -> None:
    """Run all 4 experiments and output 4 figures."""
    setup_plot_style()

    out = Path(output_dir)

    for vehicle in VEHICLES:
        print(f"Running experiment: {vehicle}")
        result = run_single_experiment(
            target_vehicle=vehicle,
            n_lanes=4,
            l_max=5,
            step=0.05,
            verbose=verbose_solver,
            n_runs=n_runs,
        )
        save_experiment_results_table(vehicle, result, out)
        plot_comparison(vehicle, result, out)

    print(f"Done. 4 figures saved to: {out.resolve()}")


if __name__ == "__main__":
    run_all_experiments(output_dir=f"experiment_figures_4lanes", verbose_solver=False, n_runs=1)
