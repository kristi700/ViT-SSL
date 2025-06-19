# 🔬 `unsupervised_evaluator.py`

Runs feature-based evaluations for self-supervised checkpoints. It can execute multiple modes in one pass:

- **`eval_knn`** – k‑NN classification on extracted features.
- **`eval_linear`** – logistic regression (linear probing).
- **`eval_umap`** – dimensionality reduction and clustering metrics with UMAP.

Call it directly to analyse a saved experiment:

```bash
python evaluators/unsupervised_evaluator.py eval.mode='eval_knn,eval_linear,eval_umap' eval.experiment_path=path/to/exp
```

The script loads the model, extracts features once, then applies the selected evaluations. Results are saved to the experiment folder.

## Automatic integration

During **DINO** training, if `eval.interval` is set in the config, the trainer will automatically call `run_evaluation` every *N* epochs. Each evaluation run is stored under the training output directory with the epoch number.