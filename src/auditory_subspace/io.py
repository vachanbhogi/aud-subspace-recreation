"""Data input/output, validation, and schema checking for auditory subspace analysis."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

# Expected Zenodo MD5 checksums (Record 18331549)
ZENODO_CHECKSUMS: Dict[str, str] = {
    "aud_ss_tools.py": "8d37ee8b2b5601adf40099311589e128",
    "cell_list.csv": "301e93a36f2fc00197e92cf7ef6718c7",
    "model_performance.csv": "cbda035628820b667d294186d232ef5a",
    "neuron_pair_similarity.csv": "fe563d361aa30be373ae27374059decf",
    "neuron_ss_tuning_pcs.csv": "946030ee9ed5376843325a33e0181efe",
    "ssrf_size_overlap.csv": "91d62e9da1c1982f04123e5145aded6d",
    "models.zip": "b22cf78a38ae25a8eef7feb77a93f371",
    "README.md": "0d23cee0d0aff76f59d61d7f22bb1a97",
    "aud_subspace_figs.ipynb": "27e2dfd88471d9b9cd64e4a1a46104d5",
    "aud_subspace_fit_demo.ipynb": "7a62d2a5784dfae9d161910f7aee3e2a",
    "recordings.zip": "3c5f5b785bc31f2cab561dfc52fee101",
    "wav.zip": "e8047588ad8b47726ade7621ab17abe8",
}


def compute_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 checksum of a file on disk."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_checksum(file_path: Path, expected_md5: Optional[str] = None) -> bool:
    """Verify that a file exists and matches its expected checksum."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    expected = expected_md5 or ZENODO_CHECKSUMS.get(file_path.name)
    if expected is None:
        raise ValueError(f"No expected checksum known for {file_path.name}")

    actual = compute_md5(file_path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {file_path.name}: expected {expected}, got {actual}"
        )
    return True


def load_config(config_path: Path | str = "config/parameters.yaml") -> Dict[str, Any]:
    """Load YAML configuration parameters."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_cell_list(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load cell_list.csv, validating primary key and basic types."""
    path = Path(raw_dir) / "cell_list.csv"
    verify_file_checksum(path)
    df = pd.read_csv(path, index_col=0)
    assert len(df) == 3259, f"Unexpected row count in cell_list: {len(df)}"
    assert "goodpred" in df.columns
    assert "siteid" in df.columns
    return df


def load_model_performance(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load model_performance.csv, validating schema."""
    path = Path(raw_dir) / "model_performance.csv"
    verify_file_checksum(path)
    df = pd.read_csv(path, index_col=0)
    assert len(df) == 3259, f"Unexpected row count in model_performance: {len(df)}"
    for col in ["LN32", "CNN32-bsg", "dfit.v95.u15", "goodpred", "siteid"]:
        assert col in df.columns, f"Missing required column: {col}"
    return df


def load_pair_similarity(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load neuron_pair_similarity.csv."""
    path = Path(raw_dir) / "neuron_pair_similarity.csv"
    verify_file_checksum(path)
    df = pd.read_csv(path, index_col=0)
    assert len(df) == 69259, f"Unexpected row count in neuron_pair_similarity: {len(df)}"
    assert "sscc" in df.columns
    assert "same_site" in df.columns
    # Add animal identifier extracted from site ID
    df["animal"] = df["siteid1"].str.slice(0, 3)
    return df


def load_ss_tuning_pcs(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load neuron_ss_tuning_pcs.csv."""
    path = Path(raw_dir) / "neuron_ss_tuning_pcs.csv"
    verify_file_checksum(path)
    df = pd.read_csv(path, index_col=0)
    assert len(df) == 11181, f"Unexpected row count in neuron_ss_tuning_pcs: {len(df)}"
    assert "asym" in df.columns
    assert "asym_cat" in df.columns
    return df


def load_ssrf_size_overlap(raw_dir: Path | str = "data/raw") -> pd.DataFrame:
    """Load ssrf_size_overlap.csv."""
    path = Path(raw_dir) / "ssrf_size_overlap.csv"
    verify_file_checksum(path)
    df = pd.read_csv(path, index_col=0)
    assert len(df) == 109, f"Unexpected row count in ssrf_size_overlap: {len(df)}"
    assert "msize" in df.columns
    assert "celltype" in df.columns
    return df
