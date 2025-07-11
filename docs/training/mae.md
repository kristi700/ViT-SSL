# 🧵 MAE

This section covers the implementation of **MAE (Masked Autoencoder)** in the ViT-SSL framework.  
It follows the original [SimMIM paper](https://arxiv.org/pdf/2111.06377).

---

## Overview

MAE performs **encoding** into a latent space with it's encoder, that can be significantly larger than the decoder due to the fact that
it only processes the non-masked patches. Later, during inference it only uses the encoder to create feature vectors.

---

## Architecture: `MAEViT`

Defined in `model.py`, this model consists of:

- A masking function that let's one deshuffle the patches
- **Positional embeddings**
- Multiple `EncoderBlocks` and `DecoderBlocks`
- A simple **MLP head** for predicting RGB pixel values of masked patches

---

## Masking: `get_and_apply_mask`

Defined in `masking.py`, this function:

- Creates masking for mask_raito percent of the patches and applies it

- Returns:
    - visible_patches: Processed patches
    - masked_indicies: Indicies of the masked patches for loss calculation
    - resolve_ids: the ids for deshuffling patches

---

## Forward Pass

```python
x, masked_indicies, resolve_ids = self.encoder_forward(x)
x = self.decoder_forward(x, resolve_ids)
return x, masked_indicies
```

The model only predicts masked patches, but the encoder only uses the visible patches.

## Loss

The loss is a pixel-wise regression loss between the predicted and ground truth pixel values of the masked patches. Also, as proposed in
the original paper, the targets are normalized pixel-wise with std and mean.

## Training: SimMIMTrainer

Implemented in trainer.py, this trainer:

- Loads input images and applies patch masking
- Supports warmup schedulers and logs training/validation metrics

## Validation

Validation follows the same logic as training, but without gradient updates. It:

- Reconstructs patches
- Measures loss
- Logs predictions for analysis or visual debugging

| Component            | File            | Role                                                          |
|----------------------|------------------|---------------------------------------------------------------|
| `MAEViT`          | `model.py`       | Vision Transformer with patch masking and pixel prediction   |
| `get_and_apply_mask`     | `masking.py`     | Random masking of input patches and target generation        |
| `MAETrainer`      | `trainer.py`     | Training/validation loop with patch reconstruction loss      |
