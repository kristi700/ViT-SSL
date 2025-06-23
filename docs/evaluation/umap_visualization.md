# 🗺️ `umap_visualization.py`

Reduces high‑dimensional features to two dimensions with UMAP and creates several plots to inspect the learned representations. It also computes clustering metrics such as silhouette score and adjusted Rand index.

The resulting images and quality reports are saved next to the evaluated experiment.

An additional helper `create_3d_umap_animation` generates a rotating GIF for a
three‑dimensional embedding. Run `scripts/3d_umap_visualizer.py` with the same
`eval_config` used for training to create the animation.