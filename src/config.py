from pathlib import Path
import os


# ============================================================
# VeReMi-Graph portable configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# External input: original VeReMi dataset
#
# Recommended:
# Set VEREMI_RAW_ROOT as an environment variable.
#
# Alternatively, edit config.example.py / local configuration.
# ------------------------------------------------------------

RAW_DATA_ROOT = Path(
    os.environ.get(
        "VEREMI_RAW_ROOT",
        PROJECT_ROOT / "raw_data"
    )
)


# ------------------------------------------------------------
# Intermediate/generated outputs
# ------------------------------------------------------------

WORK_ROOT = Path(
    os.environ.get(
        "VEREMI_WORK_ROOT",
        PROJECT_ROOT / "work"
    )
)

GRAPH_ENGINE_ROOT = WORK_ROOT / "graph_engine"

GRAPH_FEATURE_ROOT = WORK_ROOT / "graph_features"

GRAPH_DATASET_ROOT = WORK_ROOT / "graph_dataset"

SEMANTIC_VALIDATION_ROOT = WORK_ROOT / "semantic_validation"

FEATURE_AUDIT_ROOT = WORK_ROOT / "audits" / "feature_integrity"

INITIAL_SPLIT_ROOT = WORK_ROOT / "splits" / "initial"

BALANCED_SPLIT_ROOT = WORK_ROOT / "splits" / "balanced"

FEATURE_SCHEMA_ROOT = WORK_ROOT / "feature_schema"


# ------------------------------------------------------------
# Published/released VeReMi-Graph dataset
#
# Can point to a downloaded copy of the Mendeley release.
# ------------------------------------------------------------

RELEASE_ROOT = Path(
    os.environ.get(
        "VEREMI_RELEASE_ROOT",
        PROJECT_ROOT / "release"
    )
)


# ------------------------------------------------------------
# Create local generated-output directories
# ------------------------------------------------------------

for directory in [
    WORK_ROOT,
    GRAPH_ENGINE_ROOT,
    GRAPH_FEATURE_ROOT,
    GRAPH_DATASET_ROOT,
    SEMANTIC_VALIDATION_ROOT,
    FEATURE_AUDIT_ROOT,
    INITIAL_SPLIT_ROOT,
    BALANCED_SPLIT_ROOT,
    FEATURE_SCHEMA_ROOT,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


    # ============================================================
# Compatibility aliases used by the original construction code
# ============================================================

ROOT = RAW_DATA_ROOT
OUTPUT_ROOT = GRAPH_ENGINE_ROOT
WINDOW_SIZE = 1.0