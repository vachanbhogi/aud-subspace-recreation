.PHONY: env data reproduce single-site test clean

PYTHON := .venv/bin/python
RUFF := .venv/bin/ruff
PYTEST := .venv/bin/pytest

env:
	@which uv > /dev/null || (echo "uv is required: https://docs.astral.sh/uv/" && exit 1)
	@test -d .venv || uv venv --python 3.11
	@uv pip install -e ".[dev]"

data:
	$(PYTHON) scripts/01_download_and_verify.py

reproduce:
	$(PYTHON) scripts/02_reproduce.py

# Requires the optional NEMS/TensorFlow setup and data/raw/recordings.zip.
single-site:
	$(PYTHON) scripts/07_single_site_demo.py

test:
	$(RUFF) check src tests scripts
	$(RUFF) format --check src tests scripts
	$(PYTEST) -q

clean:
	rm -rf .pytest_cache .ruff_cache results
