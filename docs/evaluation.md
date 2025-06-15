# 🔍 Evaluation Scripts

This page explains the small helper scripts used to evaluate trained checkpoints and visualise their behaviour.

## `supervised_evaluator.py`

Runs inference for a **supervised ViT classifier** on a single image and overlays the final attention map of the `[CLS]` token. Useful for quickly inspecting what the model focuses on.

### Usage

```bash
python scripts/supervised_evaluator.py --checkpoint path/to/supervised.pth --image path/to/image.jpg
```

The script prints basic checkpoint info and saves `attention_visualization.png` in the current directory.

## `simmim_evaluator.py`

Loads a **SimMIM** checkpoint and visualises the masked image modelling process. It displays:

1. the original image,
2. the masked input,
3. the model reconstruction of masked patches.

### Usage

```bash
python scripts/simmim_evaluator.py --checkpoint path/to/simmim.pth --image path/to/image.jpg
```

A `simmim_reconstruction.png` file is produced showing the three-panel comparison.

These scripts rely on the configuration stored inside each checkpoint so no additional config file is required.