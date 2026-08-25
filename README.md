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
    `-- VeReMi_Graph_Demo_Explorer.ipynb