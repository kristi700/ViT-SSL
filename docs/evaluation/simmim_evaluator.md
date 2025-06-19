# 🧩 `simmim_evaluator.py`

Loads a **SimMIM** checkpoint and visualises the masked image modelling process. It displays:

1. the original image,
2. the masked input,
3. the model reconstruction of masked patches.

## Usage

```bash
python scripts/simmim_evaluator.py --checkpoint path/to/simmim.pth --image path/to/image.jpg
```

A `simmim_reconstruction.png` file is produced showing the three-panel comparison.

These scripts rely on the configuration stored inside each checkpoint so no additional config file is required.