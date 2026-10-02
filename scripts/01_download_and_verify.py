"""Download and checksum only the tables required by the core workflow."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from auditory_subspace.io import ZENODO_CHECKSUMS, compute_md5

ZENODO_FILES_URL = "https://zenodo.org/api/records/18331549/files"
CORE_FILES = (
    "cell_list.csv",
    "model_performance.csv",
    "neuron_pair_similarity.csv",
    "ssrf_size_overlap.csv",
)


def main() -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in CORE_FILES:
        target = raw_dir / filename
        expected = ZENODO_CHECKSUMS[filename]
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".download")
            try:
                url = f"{ZENODO_FILES_URL}/{filename}/content"
                print(f"Downloading {filename} from Zenodo")
                urllib.request.urlretrieve(url, temporary)
                actual = compute_md5(temporary)
                if actual != expected:
                    raise ValueError(
                        f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
                    )
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        actual = compute_md5(target)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {filename}: expected {expected}, got {actual}")
        print(f"Verified {filename}")


if __name__ == "__main__":
    main()
