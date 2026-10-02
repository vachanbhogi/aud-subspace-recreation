# auditory-subspace

Recreation and exploratory validation of auditory cortical subspace models from [Wingert et al. (Nature Neuroscience 2026)](https://doi.org/10.1038/s41593-026-02216-0).

Recompute published summaries from open Zenodo tables. Compare LN vs. CNN receptive field subspaces and verify local population encoding dimensions.

## Quickstart

Requires Python 3.11 and [`uv`](https://github.com/astral-sh/uv).

```bash
make env
make data
make reproduce
make test
```

## Workflows

- `make reproduce`: Recompute responsive unit counts, site medians, and pair metrics from raw Zenodo tables.
- `make single-site`: Refit CNN/LN models on site `CLT027c`, extract dSTRFs, and evaluate subspace overlap (requires TensorFlow + NEMS).
- `make test`: Run schema verification and regression checks.

## Data

Source dataset hosted on [Zenodo](https://zenodo.org/records/18331549). Checksums verified on fetch.
