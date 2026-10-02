"""Schema and checksum validation tests for published datasets."""

from pathlib import Path

from auditory_subspace.io import (
    load_cell_list,
    load_model_performance,
    load_pair_similarity,
    verify_file_checksum,
)


def test_raw_files_checksums():
    """Verify that all core raw files exist and have intact MD5 checksums."""
    raw_dir = Path("data/raw")
    core_files = [
        "cell_list.csv",
        "model_performance.csv",
        "neuron_pair_similarity.csv",
        "neuron_ss_tuning_pcs.csv",
        "ssrf_size_overlap.csv",
        "models.zip",
        "README.md",
        "aud_ss_tools.py",
        "aud_subspace_figs.ipynb",
        "aud_subspace_fit_demo.ipynb",
    ]
    for fname in core_files:
        path = raw_dir / fname
        assert path.exists(), f"Missing raw file: {path}"
        assert verify_file_checksum(path), f"Checksum failed for: {path}"


def test_cell_list_schema():
    """Verify cell_list.csv shape and critical columns."""
    df = load_cell_list()
    assert df.shape[0] == 3259
    assert "goodpred" in df.columns
    assert "siteid" in df.columns
    assert "sw" in df.columns
    assert "depth" in df.columns
    assert "area" in df.columns


def test_model_performance_schema():
    """Verify model_performance.csv shape and columns."""
    df = load_model_performance()
    assert df.shape[0] == 3259
    assert "LN32" in df.columns
    assert "CNN32-bsg" in df.columns
    assert "dfit.v95.u15" in df.columns


def test_pair_similarity_schema():
    """Verify neuron_pair_similarity.csv shape and within-site consistency."""
    df = load_pair_similarity()
    assert df.shape[0] == 69259
    assert "sscc" in df.columns
    assert df["same_site"].all(), "Expected all pairs in table to be within-site"
    assert "animal" in df.columns
    assert set(df["animal"].unique()) == {"CLT", "LMD", "PRN", "SLJ"}
