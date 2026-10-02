"""Regression checks for directly recomputed descriptive dataset summaries."""

import numpy as np

from auditory_subspace.io import (
    load_cell_list,
    load_model_performance,
    load_pair_similarity,
    load_ssrf_size_overlap,
)


def test_responsive_units_count():
    cells = load_cell_list()
    a1_peg = cells[cells["area"].isin(["A1", "PEG"])]
    assert (len(a1_peg), int(a1_peg["goodpred"].sum())) == (2874, 2337)


def test_a1_pair_count_under_paper_filters():
    pairs = load_pair_similarity()
    selected = (
        (pairs["area"] == "A1")
        & (pairs["r_test"] > 0.15)
        & (pairs["depthcat1"] < 800)
        & (pairs["depthcat2"] < 800)
    )
    assert int(selected.sum()) == 42140


def test_model_performance_descriptive_all_site_medians():
    performance = load_model_performance()
    site_medians = (
        performance[performance["goodpred"]]
        .groupby("siteid")[["LN32", "CNN32-bsg", "dfit.v95.u15"]]
        .median()
    )
    medians = site_medians.median()
    assert np.isclose(medians["LN32"], 0.415, atol=0.002)
    assert np.isclose(medians["CNN32-bsg"], 0.600, atol=0.002)
    assert np.isclose(medians["dfit.v95.u15"], 0.583, atol=0.002)
    assert len(site_medians) == 68


def test_published_pair_table_only_contains_within_site_pairs():
    pairs = load_pair_similarity()
    assert pairs["same_site"].all()


def test_ssrf_table_descriptive_discrepancy():
    ssrf = load_ssrf_size_overlap()
    paired = (
        ssrf[ssrf["celltype"].isin(["R", "N"])]
        .groupby(["siteid", "celltype"])["msize"]
        .mean()
        .unstack("celltype")
        .dropna()
    )
    assert len(paired) == 27
    assert np.isclose(paired["R"].mean(), 0.072, atol=0.001)
    assert np.isclose(paired["N"].mean(), 0.128, atol=0.001)
