# 🧩 SimMIM

This section covers the implementation of **SimMIM (Simple Masked Image Modeling)** in the ViT-SSL framework.  
It follows the original [SimMIM paper](https://arxiv.org/pdf/2111.09886).

---

## Overview

SimMIM performs **masked patch prediction**, where parts of an input image are hidden (masked) and the model learns to reconstruct those regions.  
The approach is conceptually similar to BERT-style pretraining but adapted for vision using **patch-level masking and pixel-level regression**.

---

## Architecture: `SimMIMViT`

Defined in `model.py`, this model consists of:

- A custom **patch embedding** via `Unfold + Linear`
- Learnable **mask token** for masked patches
- **Positional embeddings**
- Multiple `EncoderBlocks`
- A simple **MLP head** for predicting RGB pixel values of masked patches

The masked tokens are inserted directly into the sequence before encoding. No `[CLS]` token is used during pretraining.

---

## Masking: `simple_masking`

Defined in `masking.py`, this function:

- Selects a random subset of patches to mask using boolean masks
- Returns:
    - The original patches (unchanged)
    - A binary mask indicating which patches are masked
    - The target pixels for loss computation

---

## Forward Pass

```python
patches = Unfold(image)
patches, bool_mask, targets = simple_masking(patches)

encoder_input = torch.where(mask, mask_token, projected_patches)
encoder_input += pos_emb
encoded = transformer(encoder_input)

output = simmim_head(encoded[masked_positions])
```

The model only predicts masked patches, this makes training efficient and focused.

## Loss

The loss is a pixel-wise regression loss (e.g., MSE or L1) between the predicted and ground truth pixel values of the masked patches:

`loss = criterion(predicted_pixels, target_pixels)`

## Training: SimMIMTrainer

Implemented in trainer.py, this trainer:

- Loads input images and applies patch masking
- Flattens predictions and targets for loss computation
- Supports warmup schedulers and logs training/validation metrics

## Validation

Validation follows the same logic as training, but without gradient updates. It:

- Reconstructs patches
- Measures loss
- Logs predictions for analysis or visual debugging


| Component            | File            | Role                                                          |
|----------------------|------------------|---------------------------------------------------------------|
| `SimMIMViT`          | `model.py`       | Vision Transformer with patch masking and pixel prediction   |
| `simple_masking`     | `masking.py`     | Random masking of input patches and target generation        |
| `SimMIMTrainer`      | `trainer.py`     | Training/validation loop with patch reconstruction loss      |
