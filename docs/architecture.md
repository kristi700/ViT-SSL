# 🗂️ Code Structure

This page provides a high-level overview of the main components in **ViT-SSL** and how they fit together.

## `vit_core`
Core implementation of the Vision Transformer (ViT) and self-supervised variants.

- `patch_embedding.py` – splits the image into patches and projects them to the embedding dimension.
- `attention.py` – multi-head self-attention layer used inside each transformer block.
- `feed_forward.py` – MLP block that follows the attention mechanism.
- `encoder_block.py` – combines attention and feed-forward layers into a single transformer block.
- `mlp_head.py` – classification head applied to the final representation.
- `vit.py` – convenience wrapper assembling the full ViT model.
- `ssl/` – implementations of DINO and SimMIM specific components.

## `utils`
Utilities used throughout training and evaluation.

- `trainers/` – training loops for Supervised, DINO and SimMIM methods.
- `logger.py` – colored logging based on the `rich` library.
- `metrics.py` – common evaluation metrics.
- `schedulers.py` – helper learning rate schedulers.

## `configs`
[Hydra](https://hydra.cc/) configuration files. Base configs live under `configs/base/` and each method has overrides (e.g. `configs/dino/`, `configs/simmim/`). The root `config.yaml` selects which set of overrides to apply.

For details on each configurability, see the [Configuration Guide](configs.md) section.

## Other directories

- `data/` – small dataset wrappers used for experiments.
- `scripts/` – evaluation and helper scripts.
- `tests/` – PyTest unit tests for the core modules.
- `train.py` – main entry point that loads a config and launches training.

Use these modules together to experiment with different SSL approaches and fine-tuning scenarios.

For details on each algorithm, see the [Training Methods](training/index.md) section.