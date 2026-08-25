"""
Example local configuration for VeReMi-Graph.

The large datasets are not included in this code repository.

Set these paths to the locations of:

1. the original VeReMi source dataset; and
2. optionally, the published VeReMi-Graph v1.0 dataset.
"""

from pathlib import Path


RAW_DATA_ROOT = Path(
    r"C:\path\to\original\VeReMi_Dataset"
)

RELEASE_ROOT = Path(
    r"C:\path\to\VeReMi-Graph-v1.0-release"
)