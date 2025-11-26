#!/usr/bin/env python3
# plot_datagen_overview_separate.py
#
# Assumes that plot_runtime_profiling_variants.py has already produced
#   single_normal_timings_{variant}.csv
#   single_t_timings_{variant}.csv
#   multi_normal_timings_{variant}.csv
#   multi_t_timings_{variant}.csv
#   convergence_timings_{variant}.csv
# in results/tables/.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------- configuration ----------------------

VARIANTS = {
    "baseline": {
        "pretty": "Baseline",
    },
    "cholesky": {
        "pretty": "Cholesky",
    },
    "cholesky+parallelization": {
        "pretty": "Cholesky + Parallelization",
    },
}

SIM_NAMES = {
    "single": "Single simulation",
    "multi": "Multi simulation",
    "convergence": "Convergence simulation",
}

# cholesky+parallelization에서 사용한 코어 수
# (질문에서 말한 대로 세 시뮬레이션 모두 6코어)
NUM_CORES = {
    "single": {
        "baseline": 1,
        "cholesky": 1,
        "cholesky+parallelization": 6,
    },
    "multi": {
        "baseline": 1,
        "cholesky": 1,
        "cholesky+parallelization": 6,
    },
    "convergence": {
        "baseline": 1,
        "cholesky": 1,
        "cholesky+parallelization": 6,
    },
}


# ---------------------- aesthetics ----------------------

def _set_pub_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.constrained_layout.use": True,
    })


def _ensure_dirs():
    base = Path("results")
    (base / "figures").mkdir(exist_ok=True, parents=True)


def _save(fig, stem: str):
    out_pdf = Path("results/figures") / f"{stem}.pdf"
    out_svg = Path("results/figures") / f"{stem}.svg"
    fig.savefig(out_pdf)
    fig.savefig(out_svg)
    print(f"Saved: {out_pdf}\n       {out_svg}")


# ---------------------- data collection ----------------------

def _collect_datagen_stats():
    """
    각 시뮬레이션(single/multi/convergence) × variant별:
      - data generation 총 시간 (cholesky+parallelization은 코어 수로 나눈 per-core 기준)
      - 전체 runtime 대비 비중
    을 계산해서 dict로 반환.
    """
    base = Path("results/tables")

    stats = {
        "single": {vk: {} for vk in VARIANTS},
        "multi": {vk: {} for vk in VARIANTS},
        "convergence": {vk: {} for vk in VARIANTS},
    }

    for variant_key in VARIANTS.keys():
        # --- single simulation: normal + t 합산 ---
        sn = pd.read_csv(base / f"single_normal_timings_{variant_key}.csv")
        st = pd.read_csv(base / f"single_t_timings_{variant_key}.csv")

        single_data_gen = sn["data_gen"].sum() + st["data_gen"].sum()
        single_total = (
            sn[["data_gen", "computing_estimator", "principal_angles"]].to_numpy().sum()
            + st[["data_gen", "computing_estimator", "principal_angles"]].to_numpy().sum()
        )

        # --- multi simulation: normal + t 합산 ---
        mn = pd.read_csv(base / f"multi_normal_timings_{variant_key}.csv")
        mt = pd.read_csv(base / f"multi_t_timings_{variant_key}.csv")

        multi_data_gen = mn["data_gen"].sum() + mt["data_gen"].sum()
        multi_total = (
            mn[["data_gen", "computing_estimator", "principal_angles"]].to_numpy().sum()
            + mt[["data_gen", "computing_estimator", "principal_angles"]].to_numpy().sum()
        )

        # --- convergence simulation ---
        cv = pd.read_csv(base / f"convergence_timings_{variant_key}.csv")
        conv_data_gen = cv["data_gen"].sum()
        conv_total = cv[["data_gen", "computing_estimator", "principal_angles"]].to_numpy().sum()

        for sim_name, (dg_raw, tot_raw) in {
            "single": (single_data_gen, single_total),
            "multi": (multi_data_gen, multi_total),
            "convergence": (conv_data_gen, conv_total),
        }.items():
            cores = NUM_CORES[sim_name][variant_key]
            scale = 1.0 / cores  # per-core 기준 시간
            dg = dg_raw * scale
            tot = tot_raw * scale
            frac = dg / tot if tot > 0 else np.nan

            stats[sim_name][variant_key] = {
                "data_gen": dg,
                "fraction": frac,
            }

    return stats


# ---------------------- plotting ----------------------

def _plot_datagen_time(stats):
    """
    Plot 1: data generation 절대 시간 (per core for parallel)
    x축: Baseline / Cholesky / Cholesky + Parallelization
    y축: seconds
    선 3개: single / multi / convergence
    """
    methods = list(VARIANTS.keys())
    pretty_methods = [VARIANTS[m]["pretty"] for m in methods]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    sim_order = ["single", "multi", "convergence"]

    for j, sim_name in enumerate(sim_order):
        color = colors[j % len(colors)]
        y_time = [stats[sim_name][m]["data_gen"] for m in methods]

        ax.plot(
            x,
            y_time,
            marker="o",
            linestyle="-",
            color=color,
            label=SIM_NAMES[sim_name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_methods)
    ax.set_xlabel("Simulation variant")
    ax.set_ylabel("Data generation time (seconds, per core for parallel)")
    ax.set_title("Data generation time across simulations")
    ax.legend(loc="best", frameon=False)

    _save(fig, "datagen_time_overview")
    plt.close(fig)


def _plot_datagen_share(stats):
    """
    Plot 2: data generation 비율 (fraction of total runtime)
    x축: Baseline / Cholesky / Cholesky + Parallelization
    y축: share (0~1)
    선 3개: single / multi / convergence
    """
    methods = list(VARIANTS.keys())
    pretty_methods = [VARIANTS[m]["pretty"] for m in methods]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))

    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    sim_order = ["single", "multi", "convergence"]

    for j, sim_name in enumerate(sim_order):
        color = colors[j % len(colors)]
        y_frac = [stats[sim_name][m]["fraction"] for m in methods]

        ax.plot(
            x,
            y_frac,
            marker="s",
            linestyle="--",
            color=color,
            label=SIM_NAMES[sim_name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pretty_methods)
    ax.set_xlabel("Simulation variant")
    ax.set_ylabel("Data generation share of total runtime")
    ax.set_title("Data generation share across simulations")
    ax.legend(loc="best", frameon=False)

    _save(fig, "datagen_share_overview")
    plt.close(fig)


# ---------------------- main ----------------------

def main():
    _ensure_dirs()
    _set_pub_style()

    print("Collecting data-generation stats across variants...")
    stats = _collect_datagen_stats()

    print("Plotting data-generation time overview...")
    _plot_datagen_time(stats)

    print("Plotting data-generation share overview...")
    _plot_datagen_share(stats)


if __name__ == "__main__":
    main()