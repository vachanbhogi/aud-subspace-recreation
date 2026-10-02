"""Recompute descriptive benchmarks from the published summary tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from auditory_subspace.io import (
    load_cell_list,
    load_config,
    load_model_performance,
    load_pair_similarity,
    load_ssrf_size_overlap,
)
from auditory_subspace.plotting import plot_fig2_model_performance, plot_fig4_ssrf_size


def run_reproductions() -> dict[str, Any]:
    """Recompute counts and descriptive summaries without site-level inference."""
    config = load_config()
    benchmarks = config["published_benchmarks"]
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cells = load_cell_list()
    performance = load_model_performance()
    pairs = load_pair_similarity()
    ssrf = load_ssrf_size_overlap()

    a1_peg = cells[cells["area"].isin(["A1", "PEG"])]
    responsive_count = int(a1_peg["goodpred"].sum())
    site_metrics = (
        performance[performance["goodpred"]]
        .groupby("siteid")[["LN32", "CNN32-bsg", "dfit.v95.u15"]]
        .median()
    )
    all_site_medians = site_metrics.median()
    # This is a ratio of squared prediction correlations, not a direct
    # measurement of biological response variance explained.
    prediction_ratio = (performance["dfit.v95.u15"] ** 2) / (performance["CNN32-bsg"] ** 2)
    ratio_by_site = (
        prediction_ratio[performance["goodpred"]]
        .groupby(performance.loc[performance["goodpred"], "siteid"])
        .median()
    )

    ssrf_pairs = (
        ssrf[ssrf["celltype"].isin(["R", "N"])]
        .groupby(["siteid", "celltype"])["msize"]
        .mean()
        .unstack("celltype")
        .dropna()
    )
    a1_pair_filter = (
        (pairs["area"] == "A1")
        & (pairs["r_test"] > config["filters"]["r_test_min"])
        & (pairs["depthcat1"] < config["filters"]["depth_max_um"])
        & (pairs["depthcat2"] < config["filters"]["depth_max_um"])
    )

    values: dict[str, Any] = {
        "responsive_units_a1_peg": responsive_count,
        "total_units_a1_peg": len(a1_peg),
        "site_count": int(site_metrics.shape[0]),
        "ln_site_median_all": float(all_site_medians["LN32"]),
        "cnn_site_median_all": float(all_site_medians["CNN32-bsg"]),
        "subspace_site_median_all": float(all_site_medians["dfit.v95.u15"]),
        "median_squared_prediction_ratio": float(ratio_by_site.median()),
        "within_site_pair_count": int(pairs["same_site"].sum()),
        "pair_table_count": len(pairs),
        "within_site_pair_ssi_median": float(pairs["sscc"].median()),
        "ssrf_regular_mean_in_table": float(ssrf_pairs["R"].mean()),
        "ssrf_narrow_mean_in_table": float(ssrf_pairs["N"].mean()),
        "ssrf_paired_site_count_in_table": int(len(ssrf_pairs)),
        "a1_pair_count_filtered": int(a1_pair_filter.sum()),
        "a1_pair_count_all_areas": len(pairs),
        "paper_benchmarks": benchmarks,
    }

    rows = [
        {
            "quantity": "Responsive units in A1 and PEG",
            "published_value": f"{benchmarks['responsive_units_a1_peg']} / {benchmarks['total_units_a1_peg']}",
            "computed_value": f"{responsive_count} / {len(a1_peg)}",
            "interpretation": "Exact count from published cell table.",
        },
        {
            "quantity": "LN, CNN, and subspace median of site medians (all sites)",
            "published_value": "LN 0.416; CNN 0.600; subspace 0.584 (caption) / 0.585 (text)",
            "computed_value": (
                f"LN {all_site_medians['LN32']:.3f}; CNN {all_site_medians['CNN32-bsg']:.3f}; "
                f"subspace {all_site_medians['dfit.v95.u15']:.3f}; n={len(site_metrics)} sites"
            ),
            "interpretation": "Descriptive comparison of supplied model-performance table.",
        },
        {
            "quantity": "Median squared subspace-to-CNN prediction-correlation ratio",
            "published_value": "Approximately 0.954",
            "computed_value": f"{ratio_by_site.median():.3f}",
            "interpretation": "Prediction-correlation ratio; not a direct response-variance measurement.",
        },
        {
            "quantity": "A1 pair count under paper thresholds",
            "published_value": str(benchmarks["a1_pairs_count"]),
            "computed_value": str(int(a1_pair_filter.sum())),
            "interpretation": "Count from pair table using A1, r_test, and depth filters.",
        },
        {
            "quantity": "Regular/narrow SSRF area means in supplied overlap table",
            "published_value": "Regular 0.076; narrow 0.115; paper reports n=39",
            "computed_value": (
                f"Regular {ssrf_pairs['R'].mean():.3f}; narrow {ssrf_pairs['N'].mean():.3f}; "
                f"paired sites in table n={len(ssrf_pairs)}"
            ),
            "interpretation": "Table-only descriptive comparison; discrepancy requires clarification.",
        },
        {
            "quantity": "Within-site pair SSI versus between-site comparison",
            "published_value": "Within-site 0.55; between-site 0.35",
            "computed_value": (
                f"Within-site pair median {pairs['sscc'].median():.3f} from "
                f"{int(pairs['same_site'].sum())}/{len(pairs)} within-site pairs; "
                "between-site comparison unavailable in supplied pair table"
            ),
            "interpretation": "The supplied pair table cannot reproduce the paper's unit-versus-site comparison.",
        },
    ]

    pd.DataFrame(rows).to_csv(results_dir / "reproduction_table.csv", index=False)
    (results_dir / "reproduction_metrics.json").write_text(json.dumps(values, indent=2) + "\n")
    plot_fig2_model_performance(performance, figures_dir / "fig2_model_performance.png")
    plot_fig4_ssrf_size(ssrf, figures_dir / "fig4_ssrf_size.png")
    print("Wrote descriptive outputs to results/.")
    return values


if __name__ == "__main__":
    run_reproductions()
