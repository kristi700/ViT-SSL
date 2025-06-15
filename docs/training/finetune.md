# 🔧 Fine-tuning Workflow

Fine-tuning adapts a pretrained ViT model to a new dataset. The pipeline reuses the supervised trainer but loads weights from a previous run.

---

## Overview

1. **Select mode** – set `training.type: "finetune"` in the config.
2. **Load checkpoint** – provide `training.pretrained_path` pointing to the saved model.
3. **Freeze layers** – if `training.freeze_backbone: true`, the encoder is frozen and only the head is trained.
4. **Check weights** – `load_pretrained_model()` matches shapes and logs any mismatches.
5. **Train** – `SupervisedTrainer` handles the cross‑entropy loop and logs metrics.

The workflow allows initializing from SimMIM, DINO or any compatible checkpoint.

---

## Typical Steps

```bash
# Choose the finetune config
python train.py training.type=finetune training.pretrained_path=/path/to/checkpoint.pth
```

This can be set in the config for ease of use, as explained in the [Configuration Guide](configs.md)

During startup the script:

- Builds a `ViT` model with the classification head.
- Loads the pretrained weights using `load_pretrained_model()`.
- Optionally freezes backbone parameters.
- Runs `SupervisedTrainer.fit()` for the specified number of epochs.

---

## Key Components

| Component | File | Role |
|-----------|------|------|
| `load_pretrained_model` | `train.py` | Loads weights & handles shape mismatches |
| `freeze_backbone` | `train.py` | Stops gradient updates for encoder blocks |
| `SupervisedTrainer` | `utils/train_utils.py` | Training loop used for fine‑tuning |

Fine-tuning is thus simply supervised training initialized from a checkpoint, making it easy to adapt SSL models to downstream tasks.