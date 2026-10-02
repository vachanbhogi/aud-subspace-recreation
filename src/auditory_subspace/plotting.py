"""Plot descriptive summaries from the published auditory-subspace tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_style() -> None:
    """Apply consistent styles to generated summary plots."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )


def plot_fig2_model_performance(
    mp_df: pd.DataFrame,
    out_path: Path | str = "results/figures/fig2_model_performance.png",
) -> None:
    """Plot selected model-performance summaries from the supplied table."""
    setup_style()
    f, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # Responsive units
    gp = mp_df[mp_df["goodpred"]].copy()
    site_gp = gp.groupby("siteid")[["LN32", "CNN32-bsg", "dfit.v95.u15"]].median()

    # Panel A: LN vs CNN
    ax = axes[0]
    ax.scatter(
        gp["LN32"], gp["CNN32-bsg"], s=2, color="lightgray", alpha=0.5, label=f"Units (n={len(gp)})"
    )
    ax.scatter(
        site_gp["LN32"],
        site_gp["CNN32-bsg"],
        s=10,
        color="black",
        label=f"Sites (n={len(site_gp)})",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.75)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("LN Prediction (r)")
    ax.set_ylabel("CNN Prediction (r)")
    ax.set_title("LN vs. CNN Performance")
    ax.legend(loc="lower right", frameon=False, fontsize=6)

    # Panel B: SS vs CNN
    ax = axes[1]
    ax.scatter(gp["dfit.v95.u15"], gp["CNN32-bsg"], s=2, color="lightgray", alpha=0.5)
    ax.scatter(site_gp["dfit.v95.u15"], site_gp["CNN32-bsg"], s=10, color="black")
    ax.plot([0, 1], [0, 1], "k--", lw=0.75)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("Subspace (v95) Prediction (r)")
    ax.set_ylabel("CNN Prediction (r)")
    ax.set_title("Subspace vs. CNN Performance")

    # Panel C: squared prediction-correlation ratios relative to CNN
    ax = axes[2]
    var_df = gp.copy()
    var_df["LN32_ratio"] = (var_df["LN32"] ** 2) / (var_df["CNN32-bsg"] ** 2)
    var_df["SS_ratio"] = (var_df["dfit.v95.u15"] ** 2) / (var_df["CNN32-bsg"] ** 2)
    site_ratio = var_df.groupby("siteid")[["LN32_ratio", "SS_ratio"]].median().clip(0, 1.3)

    sns.stripplot(data=site_ratio, color="gray", jitter=0.2, size=3, ax=ax)
    sns.boxplot(data=site_ratio, color="white", showfliers=False, width=0.4, ax=ax)
    ax.axhline(1.0, color="black", linestyle="--", lw=0.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LN", "Subspace (95%)"])
    ax.set_ylabel("Squared prediction-correlation ratio")
    ax.set_ylim(0.0, 1.3)
    ax.set_title("Prediction metric relative to CNN")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_fig4_ssrf_size(
    ssrf_df: pd.DataFrame,
    out_path: Path | str = "results/figures/fig4_ssrf_size.png",
) -> None:
    """Plot supplied SSRF-size summaries for paired cell types by site."""
    setup_style()
    f, ax = plt.subplots(figsize=(3.0, 2.8))

    d = (
        ssrf_df[ssrf_df["celltype"].isin(["R", "N"])]
        .groupby(["siteid", "celltype"])["msize"]
        .mean()
        .unstack(-1)
        .dropna()
    )

    ax.scatter(d["R"], d["N"], s=20, color="black")
    ax.plot([0, 0.3], [0, 0.3], "k--", lw=0.75)
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 0.25)
    ax.set_xlabel("Mean SSRF Size (Regular Spiking)")
    ax.set_ylabel("Mean SSRF Size (Narrow Spiking)")
    ax.set_title(f"SSRF Size: Regular vs. Narrow\n(n={len(d)} sites)")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
