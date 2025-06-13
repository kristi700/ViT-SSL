# Welcome to ViT-SSL  👾

> A PyTorch framework for Self-Supervised and Supervised Learning with Vision Transformers.

ViT-SSL is an educational project designed to demonstrate the core ideas of **Self-Supervised Learning (SSL)** through modular, readable, and reproducible code.  
It’s currently under development and aims to make SSL concepts more accessible to developers and researchers.

This site contains detailed documentation of the project’s structure — explaining **what each module does**, **how components interact**, and **why certain design decisions were made**.

---

## ✨ Overview

**ViT-SSL** provides: <br>
- Implementations of modern SSL methods: **DINO**, **SimMIM**, and **Supervised** baselines  
- Customizable training pipelines built with **PyTorch**  
- Clear config-driven design for experimentation  
- Metrics and logging tailored for representation learning

---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kristi700/ViT-SSL.git
    cd ViT-SSL
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Quick Start

Training is unified under a single entry point: **`train.py`**.  
To switch between methods (e.g., Supervised, DINO, SimMIM, Fine-tuning), simply set the `training.type` field in the config file.

📄 For more on available config options, see [Configuration Guide](configs.md).

---
