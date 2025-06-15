# 🛠️ Configuration Guide

## ⚙️ Configuration Structure

The project uses [Hydra](https://hydra.cc/) for modular, hierarchical configuration management. Configs are split into **base definitions** and **method-specific overrides** (DINO, SimMIM, Supervised, Fine-tuning).

---

### 📁 Directory Structure

<pre><code class="language-text"> 
configs/
├── base/ # Shared defaults for all training types 
│ ├── data.yaml 
│ ├── model.yaml 
│ └── training.yaml 
├── dino/ # DINO-specific overrides 
│ ├── data.yaml 
│ ├── model.yaml 
│ ├── training.yaml 
│ ├── globals.yaml 
│ ├── locals.yaml 
│ └── metrics.yaml 
├── simmim/ # SimMIM-specific overrides 
├── supervised/ # Supervised-specific overrides 
├── finetune/ # Fine-tuning-specific overrides 
├── dino.yaml # Hydra composition entrypoint for DINO 
├── simmim.yaml 
├── supervised.yaml 
├── finetune.yaml 
└── config.yaml # Root selector that loads a full training config </code></pre>

---

### 🧱 `base/`: Shared Configs

Used by all training types to define common logic:

- **`data.yaml`**: dataset name, directory, val split, image size, etc.
- **`model.yaml`**: patch size, embedding dim, block/head count, etc.
- **`training.yaml`**: generic hyperparameters, LR schedulers, optimizers

---

### 🧩 Method-Specific Overrides

Each training strategy has its own folder (e.g. `dino/`, `simmim/`) that overrides or extends the base values.

Example from `dino.yaml`:

```yaml
defaults:
  - config_schema

  - base@data: data
  - base@model: model
  - base@training: training

  - dino@data: data
  - dino@model: model
  - dino@training: training
  - dino@transforms.globals: globals
  - dino@transforms.locals: locals
  - dino@metrics: metrics

hydra:
  run:
    dir: ./experiments/${training.type}/${now:%Y-%m-%d_%H_%M_%S}
```

---

🔁 Root Composition (config.yaml)

Sets the active method (e.g. DINO, SimMIM):

```yaml
defaults:
  - dino.yaml
  - _self_
```

To switch training modes, change dino.yaml to simmim.yaml, finetune.yaml, etc.

---

📂 Output Directory

Each run saves outputs in a method+timestamp-based directory:

`./experiments/training.type/timestamp/`

This structure keeps logs, configs, and checkpoints isolated per run.

---