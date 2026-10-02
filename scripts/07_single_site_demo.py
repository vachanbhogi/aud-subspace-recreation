"""Refit the authors' example encoding models on one raw recording site.

The workflow follows the preprocessing, architecture, fit settings, and dSTRF
sampling in ``data/raw/aud_subspace_fit_demo.ipynb``. Fits are stochastic and
are compared with (not tuned to) the published per-unit performance table.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nems.metrics import correlation
from nems.models import CNN, LN
from nems.preprocessing.normalization import log_compress
from nems.tools.recording import average_away_epoch_occurrences, load_recording
from sklearn.decomposition import PCA


def extract_site_recording(zip_path: Path, siteid: str, dest_dir: Path) -> Path:
    """Extract the matching site archive from the authors' recordings bundle."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if siteid in name and name.endswith((".tgz", ".tar.gz"))
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one recording archive for {siteid}, found {len(matches)}"
            )
        archive.extract(matches[0], dest_dir)
    return dest_dir / matches[0]


def fit_lbhb(
    model: Any,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    learning_rate: float = 1e-4,
    epochs: int = 8000,
    early_stopping_tolerance: float = 1e-4,
) -> Any:
    """Apply the two-stage coarse-to-fine routine shown in the authors' notebook."""
    model = model.sample_from_priors()
    first_options = {
        "cost_function": "nmse",
        "early_stopping_tolerance": early_stopping_tolerance * 10,
        "validation_split": 0,
        "learning_rate": learning_rate * 10,
        "epochs": epochs // 2,
    }
    second_options = {
        "cost_function": "nmse",
        "early_stopping_tolerance": early_stopping_tolerance,
        "validation_split": 0,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }
    model.layers[-1].skip_nonlinearity()
    model = model.fit(
        input=X, target=Y, backend="tf", fitter_options=first_options, batch_size=None
    )
    model.layers[-1].unskip_nonlinearity()
    model = model.fit(
        input=X,
        target=Y,
        backend="tf",
        fitter_options=second_options,
        batch_size=None,
        verbose=0,
    )
    ymin = Y.min(axis=tuple(range(Y.ndim - 1)))
    ymax = Y.max(axis=tuple(range(Y.ndim - 1)))
    margin = (ymax - ymin) * 0.1
    model.out_range = [ymin - margin, ymax + margin]
    return model


def run_single_site_demo(
    siteid: str = "CLT027c",
    recordings_zip: Path = Path("data/raw/recordings.zip"),
    processed_dir: Path = Path("data/processed/recordings"),
    results_dir: Path = Path("results"),
) -> dict[str, Any]:
    """Refit LN/CNN models, extract dSTRFs, and compare with published outputs."""
    if not recordings_zip.exists():
        raise FileNotFoundError(
            f"Missing {recordings_zip}; download it from the linked Zenodo record first."
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    archive_path = extract_site_recording(recordings_zip, siteid, processed_dir)
    recording = load_recording(str(archive_path))
    recording["stim"] = recording["stim"].rasterize().transform(log_compress, "stim")
    recording["stim"] = recording["stim"].normalize("minmax")
    recording["resp"] = recording["resp"].rasterize().normalize("minmax")

    epoch_regex = "^STIM_"
    est, val = recording.split_using_epoch_occurrence_counts(epoch_regex=epoch_regex)
    est = average_away_epoch_occurrences(est, epoch_regex=epoch_regex)
    val = average_away_epoch_occurrences(val, epoch_regex=epoch_regex)
    X_est = np.moveaxis(est["stim"].extract_epoch("REFERENCE"), 1, 2)
    Y_est = np.moveaxis(est["resp"].extract_epoch("REFERENCE"), 1, 2)
    X_val = np.moveaxis(val["stim"].extract_epoch("REFERENCE"), 1, 2)
    Y_val = np.moveaxis(val["resp"].extract_epoch("REFERENCE"), 1, 2)

    cellids = list(est["resp"].chans)
    cell_count = len(cellids)
    fs = est["resp"].fs
    f_min, f_max = 200, 20000
    shared = {
        "channels_in": 32,
        "channels_out": cell_count,
        "rank": 50,
        "share_tuning": True,
        "gaussian": False,
        "fs": fs,
        "stride": 1,
        "regularizer": "l2:4",
        "f_min": f_min,
        "f_max": f_max,
    }

    ln0 = LN.LN_pop(name=f"LN_{siteid}", time_bins=20, **shared)
    ln0.meta["cellids"] = cellids
    ln = fit_lbhb(ln0, X_est, Y_est)
    ln_r = np.asarray(correlation(ln.predict(X_val, batch_size=X_val.shape[0]), Y_val))

    cnn0 = CNN.CNN_pop(name=f"CNN_{siteid}", time_bins=15, L2=60, L1_reps=2, **shared)
    cnn0.meta["cellids"] = cellids
    cnn = fit_lbhb(cnn0, X_est, Y_est)
    cnn_r = np.asarray(correlation(cnn.predict(X_val, batch_size=X_val.shape[0]), Y_val))

    performance = pd.read_csv("data/raw/model_performance.csv", index_col=0)
    published = performance.loc[performance["siteid"] == siteid]
    common = [cellid for cellid in cellids if cellid in published.index]
    if not common:
        raise ValueError(f"No units at {siteid} matched the published performance table")
    indices = np.array([cellids.index(cellid) for cellid in common])
    published_ln = published.loc[common, "LN32"].to_numpy(dtype=float)
    published_cnn = published.loc[common, "CNN32-bsg"].to_numpy(dtype=float)
    refit_ln = ln_r[indices]
    refit_cnn = cnn_r[indices]

    # Reproduce the authors' dSTRF sampling and PCA workflow as a method check.
    stimulus = np.reshape(X_est, (X_est.shape[0] * X_est.shape[1], -1))
    lag_count = 20
    time_indices = np.arange(lag_count, stimulus.shape[0], 101)
    dstrf = cnn.dstrf(
        stimulus,
        D=lag_count,
        out_channels=np.arange(cell_count),
        t_indexes=time_indices,
        reset_backend=False,
    )["input"]
    explained_by_unit = []
    for unit_index in range(cell_count):
        features = dstrf[unit_index].reshape(dstrf.shape[1], -1)
        mean = features.mean(axis=0, keepdims=True)
        # Match the authors' notebook formula exactly; do not substitute a
        # different denominator or silently restore filtered samples.
        noise_score = np.std(features - mean, axis=1) / np.std(mean)
        clean = features[noise_score <= 5]
        if len(clean) < 5:
            raise ValueError(
                f"Unit {unit_index} has {len(clean)} dSTRFs after the authors' "
                "SNR filter; PCA with five components cannot be fit."
            )
        pca = PCA(n_components=5).fit(clean)
        components = pca.components_.copy()
        components[components.sum(axis=1) < 0] *= -1
        explained_by_unit.append(pca.explained_variance_ratio_)

    result = {
        "site_id": siteid,
        "matched_units": len(common),
        "fit_units": cell_count,
        "published_table_medians": {
            "LN32": float(np.median(published_ln)),
            "CNN32-bsg": float(np.median(published_cnn)),
        },
        "refit_medians": {
            "LN": float(np.median(refit_ln)),
            "CNN": float(np.median(refit_cnn)),
        },
        "median_refit_minus_published": {
            "LN": float(np.median(refit_ln) - np.median(published_ln)),
            "CNN": float(np.median(refit_cnn) - np.median(published_cnn)),
        },
        "unitwise_correlation_refit_vs_published": {
            "LN": float(np.corrcoef(refit_ln, published_ln)[0, 1]),
            "CNN": float(np.corrcoef(refit_cnn, published_cnn)[0, 1]),
        },
        "fraction_refit_CNN_ge_LN": float(np.mean(refit_cnn >= refit_ln)),
        "dstrf": {
            "lag_bins": lag_count,
            "sampled_timepoints": int(len(time_indices)),
            "pca_components_per_unit": 5,
            "median_cumulative_variance_top4": float(
                np.median(np.sum(np.asarray(explained_by_unit)[:, :4], axis=1))
            ),
        },
        "protocol": "Authors' aud_subspace_fit_demo.ipynb settings; stochastic refit, no tuning to published values.",
        "note": "A refit comparison is not an exact reproduction of the authors' saved model fits.",
    }
    (results_dir / "single_site_demo.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_single_site_demo()
