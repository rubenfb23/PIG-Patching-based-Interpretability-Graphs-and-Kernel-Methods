# PIG Examples

This directory contains example scripts and artifacts for the PIG (Patching-based Interpretability Graphs) framework.

## Scripts

- `run_pipeline.py`: An end-to-end example demonstrating the full pipeline from model loading to kernel classification (classical + quantum). It generates the following visualizations:
  - `heatmap_*.png`: Average patch effect heatmaps for each slice.
  - `pca_embeddings.png`: PCA projection of the graph embeddings.

## Usage

Make sure you have installed the package first:

```bash
pip install -e .
```

Then run the example:

```bash
python examples/run_pipeline.py
```
