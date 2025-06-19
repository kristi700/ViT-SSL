# 🧐 `supervised_evaluator.py`

Runs inference for a **supervised ViT classifier** on a single image and overlays the final attention map of the `[CLS]` token. Useful for quickly inspecting what the model focuses on.

## Usage

```bash
python scripts/supervised_evaluator.py --checkpoint path/to/supervised.pth --image path/to/image.jpg
```

The script prints basic checkpoint info and saves `attention_visualization.png` in the current directory.