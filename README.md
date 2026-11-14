# VeReMi-Graph v1.0 — Construction and Reproducibility Code

This repository contains the construction, validation, packaging, and
reproducibility code associated with:

**VeReMi-Graph v1.0: A Temporal Attributed Graph Dataset for Vehicular
Misbehavior Detection**

The published dataset is available from Mendeley Data:

**DOI:** https://doi.org/10.17632/s6vvgcnsvz.1

VeReMi-Graph v1.0 is a graph-structured derivative of the original
Vehicular Reference Misbehavior (VeReMi) dataset. The transformation converts
the original per-vehicle simulation logs into directed, attributed temporal
communication graphs suitable for graph-based and temporal vehicular
misbehavior-detection research.

---

## Dataset Scale

The published VeReMi-Graph v1.0 release contains:

- 225 source simulation archives
- 22,500 one-second temporal graph snapshots
- 2,016,487 graph-node instances
- 39,136,291 directed message-level edges
- 26 frozen node-input features
- 23 frozen edge-input features
- 13 preceding-snapshot graph-context features
- archive-disjoint training, validation, and test partitions
- directly loadable PyTorch Geometric graph objects

The benchmark split contains:

- 157 training archives
- 34 validation archives
- 34 test archives

Archives are mutually exclusive across the three partitions.

---

## Purpose of This Repository

The repository provides the code used to reconstruct and validate the graph
representation rather than a proposed graph-neural-network detection model.

The construction pipeline preserves:

- simulation-archive provenance;
- vehicle identity;
- sender and receiver identity;
- message-level communication events;
- temporal ordering;
- message-level ground truth;
- causal historical information;
- archive-disjoint benchmark partitions.

No proximity, k-nearest-neighbor, feature-similarity, or other synthetic
graph edges are introduced. A communication edge represents an observed
message reception in the original VeReMi logs.

---

## Repository Structure

```text
VeReMi-Graph-Code-v1.0/
|
|-- README.md
|-- LICENSE.txt
|-- requirements.txt
|-- config.example.py
|
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- veremi_graph_engine.py
|   `-- feature_engine.py
|
|-- notebooks/
|   |-- 00_semantic_validation.ipynb
|   |-- 01_build_graphs.ipynb
|   |-- 02_feature_engineering.ipynb
|   |-- 03_causal_node_state_completion.ipynb
|   |-- 04_feature_integrity_audit.ipynb
|   |-- 05_archive_split.ipynb
|   |-- 06_balanced_split_optimizer.ipynb
|   |-- 07_model_safe_schema_audit.ipynb
|   |-- 08_release_packaging.ipynb
|   `-- 09_pyg_conversion.ipynb
|
|-- analysis/
|   `-- dataset_characterization.ipynb
|
`-- examples/
    |-- load_veremi_graph.py
    `-- VeReMi_Graph_Demo_Explorer.ipynb## Reproducibility and Software Environment

The original VeReMi-Graph v1.0 release corresponds to commit
`c874d99cabdfd2b4e7207efae84ac656f4af06d8`.

This post-v1.0 reproducibility update restores the original
graph-construction configuration parameters and provides an
expanded software environment record.

### Software environment

The verified benchmark environment uses Python 3.14.6.

Principal dependencies are pinned in `requirements.txt`.
The complete installed package inventory is provided in
`environment_reproducibility.txt`.

### Source data

Obtain the original VeReMi source archives from the original
dataset provider. Place the `.tgz` files in a local directory.

Configure the source directory using `VEREMI_RAW_ROOT`.

### Construction workflow

The numbered notebooks in `notebooks/` document the following stages:

- 00: Semantic validation
- 01: Temporal graph construction
- 02: Feature engineering
- 03: Causal node-state completion
- 04: Feature-integrity auditing
- 05: Archive-disjoint splitting
- 06: Balanced split optimization
- 07: Model-safe schema verification
- 08: Release packaging
- 09: PyTorch Geometric conversion

The graph-construction configuration uses one-second windows,
ground-truth matching tolerance of 1e-6, per-archive exports,
and Zstandard-compressed Parquet files.

### Computational benchmark

The initial graph-construction stage was benchmarked on Windows 11,
using an Intel Core i7-13700 processor and approximately 16 GB RAM.

Processing all 225 VeReMi source archives required 1,278.49 seconds
(21.31 minutes), with approximate peak sampled process-tree RSS
of 2,175.39 MiB (2.12 GiB).

All 225 archives completed successfully, with zero reported
archive-level failures.

These measurements exclude subsequent feature engineering,
splitting, validation, and release packaging.

## PyTorch Geometric Training Smoke Test

A lightweight training smoke test is provided in `examples/pyg_training_smoke_test.py` to verify that the released demonstration graphs support PyTorch Geometric loading, mini-batching, forward propagation, loss computation, and backpropagation.

Download the demonstration graphs from the [VeReMi-Graph v1.0 release](https://github.com/masoodifahim/VeReMi-Graph-Code/releases/tag/v1.0) and place `small.pt`, `medium.pt`, and `large.pt` inside a `pyg/` directory at the repository root.

Run:

```bash
python examples/pyg_training_smoke_test.py
```

**Verified environment:** Python 3.13.14, NumPy 2.5.4, PyTorch 2.15.0+cpu, and PyTorch Geometric 2.8.1.

**Verified execution (CPU):**

```text
Loaded small: nodes=12, edges=98
Loaded medium: nodes=43, edges=359
Loaded large: nodes=209, edges=5752
Batch 1: graphs=2, edges=457, loss=50.960121
Batch 2: graphs=1, edges=5752, loss=9.158440
PASS: one epoch, 2 batches, 6209 edges, elapsed=0.68s
Versions: PyTorch=2.15.0+cpu; PyG=2.8.1
```

This smoke test verifies software functionality only; it is not a predictive-performance benchmark. Runtime and loss values may vary across environments.

### Environment Compatibility

The versions in `requirements.txt` reproduce the verified dataset-construction environment documented in the manuscript. They are not intended as a universal compatibility specification.

The released demonstration graphs have also been successfully tested in a separate CPU-only environment using Python 3.13.14, NumPy 2.5.4, PyTorch 2.15.0, and PyTorch Geometric 2.8.1.

Compatibility of the complete construction pipeline with Python 3.10/3.11 or other PyTorch 2.x configurations has not been independently verified. Users working in alternative environments should install mutually compatible dependencies and run the provided validation and training tests before relying on the outputs.
