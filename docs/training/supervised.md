# 🧠 Supervised Learning

This section documents the supervised classification pipeline using Vision Transformers in **ViT-SSL**.

---

## 🧱 Architecture: `ViT`

The model is implemented in `model.py` and follows the original ViT structure:

- **Patch Embedding** using a convolutional or manual unfold-based projector
- **Transformer Encoder**: stacked `EncoderBlock`s
- **Classification Head**: a LayerNorm + Linear projection (`MLPHead`) applied to the `[CLS]` token

### 🔷 Components:
- `ConvolutionalPatchEmbedding`: Conv2D patch tokenizer
- `ManualPatchEmbedding`: Alternative unfold + linear patching
- `EncoderBlock`: Multi-head attention block
- `MLPHead`: Classifier head

### 🔁 Forward Flow:

```python
x = patch_embedding(image)
x = encoder_blocks(x)
cls_token = x[:, 0]
logits = classification_head(cls_token)
```

## ⚙️ Patch Embedding Variants

Two types of patch embedding are supported:

| Name                        | Description                             |
|-----------------------------|-----------------------------------------|
| `ConvolutionalPatchEmbedding` | Efficient Conv2D-based tokenization     |
| `ManualPatchEmbedding`        | Manual Unfold + Linear patch projection |


Each variant prepends a learnable `[CLS]` token and adds a positional embedding.

## 🧠 Classification Head: MLPHead
- A lightweight head that normalizes and linearly maps the CLS token to the number of classes.
- Can be replaced with a deeper head if needed for finetuning.

##  📉 Supervised Training: SupervisedTrainer

Defined in trainer.py, this trainer handles:

- Cross-entropy loss training loop
- Accuracy tracking
- Scheduler warmup
- Backbone freezing/unfreezing for transfer learning or finetuning

##  🔁 Training Flow:

```python
for (inputs, labels) in train_loader:
    preds = model(inputs)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()
```

## 🔁 Finetuning Support

If using pre-trained weights (e.g., from SimMIM or DINO), the system supports:

- Loading and matching pretrained weights with current model
- Selective layer skipping (e.g., skipping simmim_head or mask_token)
- Positional embedding interpolation if patch count differs

This is handled via the `load_pretrained_model()` utility.

🧩 Component Summary

| Component                   | File                  | Role                                               |
|----------------------------|------------------------|----------------------------------------------------|
| `ViT`                      | `model.py`             | Full supervised ViT model                          |
| `ConvolutionalPatchEmbedding` | `patch_embedding.py` | Conv2D-based patch tokenizer                        |
| `ManualPatchEmbedding`     | `patch_embedding.py`   | Linear + Unfold tokenizer                          |
| `MLPHead`                  | `mlp_head.py`          | LayerNorm + Linear classifier                      |
| `SupervisedTrainer`        | `trainer.py`           | Cross-entropy training loop with scheduler support |
| `load_pretrained_model`    | `utils/train_utils.py` | Weight loader with shape/positional fixups         |
