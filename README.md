# 🚧 Project Status - Work in progress

# ViT-SSL
> A PyTorch Framework for Self-Supervised and Supervised Learning with Vision Transformers.

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/kristi700/ViT-SSL/blob/main/LICENSE.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Config: Hydra](https://img.shields.io/badge/Config-Hydra-89B83A.svg)](https://hydra.cc/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

## 📚 Documentation

👉 [https://kristi700.github.io/ViT-SSL](https://kristi700.github.io/ViT-SSL)

# GIF - SimMIM reconstuction, DINO attention viz

## ✨ Overview

**Vit-SSL**  is an educational project currently in development, designed to teach the principles and practice of Self-Supervised Learning (SSL). It offers intuitive explanations, reproducible code, and hands-on experiments.

## 🔑 Key Features

*   **Multiple Pre-training Methods:** Out-of-the-box support for:
    *   **DINO:** Self-distillation with no labels.
    *   **SimMIM:** A simple masked image modeling approach.
    *   **Supervised:** A standard supervised baseline for comparison.
*   **Fine-tuning:** Easily load any pre-trained checkpoint and fine-tune it on a downstream classification task.
*   **Unified Trainer Architecture:** A modular and extensible design that simplifies adding new models, datasets, or training methods.
*   **Transparent Logging:** Powered by `rich` for clear, color-coded, and organized console output.
*   **Automatic Evaluation:** Each N epcohs runs, specified evaluators (KNN Classification, Linear Probing, UMAP)

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

## 🚀 Quick Start

There is one unified training file (train.py), where one can train Supervised, DINO, SimMIM and fine-tune pre-trained models. The type of training needs to be specified in the configs.

# GIF - Showing training logger

## 📊 Results

| Pre-training Method | Fine-tuning Dataset | Top-1 Accuracy |
|---------------------|---------------------|----------------|
| DINO                | STL10               | TODO           |
| SimMIM              | STL10               | TODO           |
| Supervised          | STL10               | TODO           |

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/kristi700/ViT-SSL/blob/main/LICENSE.md) file for details.
