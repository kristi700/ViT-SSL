# 📏 `supervised_dataset_evaluator.py`

Evaluates a trained supervised ViT on an entire dataset. The script loads the configuration stored inside the checkpoint, builds the dataloader and reports classification metrics.

## Usage

```bash
python evaluators/supervised_dataset_evaluator.py checkpoint=path/to/model.pth
```

Predictions and the computed Top‑1 accuracy are written to `evaluation_results.json` and `predictions.csv` in the current directory. Pass `eval.confusion_matrix=true` in the Hydra command to additionally store a confusion matrix.