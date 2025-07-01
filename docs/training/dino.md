# 🦖 DINO

The DINO implementation in **ViT-SSL** is modular and educational by design. It follows the original [DINO paper](https://arxiv.org/pdf/2104.14294).

---

## Overview

DINO (Self-**Di**stillation with **No** Labels) uses a **teacher–student framework** to learn representations without supervision. The teacher is an EMA (exponential moving average) of the student. Training aims to align their output distributions using softmax cross-view consistency.

---

## Architecture

- Defined in `vit_core/ssl/dino/model.py`
- Composed of:
    - `ViTBackbone`: shared ViT encoder used for both student and teacher
    - `DINOHead`: MLP head with optional normalization and projection
- Initializes:
    - `student_backbone` ← trainable
    - `teacher_backbone` ← frozen copy, updated with momentum
    - Separate heads for student and teacher
    - Center buffer for output normalization (Eq. 4 in the paper)

```python
teacher_output = teacher_head(teacher_backbone(x))
student_output = student_head(student_backbone(x))
```

Teacher outputs are updated via momentum_update_teacher() using a scheduled momentum.

### Forward Pass

```python
def forward(multi_crop_views, num_global_views):
    student_input = torch.cat(all_views)
    teacher_input = torch.cat(global_views)

    student_output = student(student_input)
    teacher_output = teacher(teacher_input)

    return teacher_output, student_output
```


### Loss: DINOLoss
- Implements *Equation 1* from the DINO paper
- Applies temperature scaling + centering to the teacher logits
- Uses cross-view prediction: student tries to predict teacher output from different views
- Cross entropy is computed between softmaxed teacher and log-softmaxed student outputs:

`loss = -(softmax(teacher) * log_softmax(student)).sum().mean()` <br>

*Defined in vit_core/ssl/dino/loss.py*

### Training: DINOTrainer
- Inherits from a generic BaseTrainer
- Implements:
    - create_criterion(): builds the DINOLoss
    - train_epoch(): training logic, view reshaping, loss calc, teacher update, warmup
    - validate(): similar logic without gradient computation

- Highlights
    - Teacher momentum is scheduled with a cosine scheduler via DINOMomentumScheduler
    - Teacher temperature can be scheduled (cosine or linear) with DINOTeacherTempScheduler
    - Both teacher and student outputs are reshaped per view before computing the loss
    - Centering is updated at every step as per DINO's original formulation
    - Uses variable shaped inputs for local and global views

### Modular Design

| Component  | File | Role |
|------------------------|----------------|-------------------------------------|
| `DINOViT`  | `model.py` | Dual backbone + head w/ EMA update |
| `DINOHead` | `head.py`  | Nonlinear projection head |
| `DINOLoss`   | `loss.py`  | Self-distillation loss|
| `DINOMomentumScheduler`| `dino_utils.py`| Momentum scheduler |
| `DINOTrainer` | `trainer.py`| Full training loop |
