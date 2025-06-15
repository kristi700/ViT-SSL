# 📊 Data Format

This page describes the lightweight dataset wrappers under [`data/datasets.py`](../data/datasets.py) and how input files are organized.

---

## `CIFAR10Dataset`

- **CSV annotations** – two columns: image filename **without extension** and label.
- **Image folder** – `root_dir` contains the corresponding `*.png` files.

```
images/
  00001.png
  00002.png
...
train.csv
```

Each row in `train.csv` contains `image_name,label`. Labels are mapped to indices using the unique values discovered in the file.

## `STL10Dataset`

- **JSON annotations** – similar two-field structure `["file","label"]`.
- **Image folder** – `root_dir` points to the directory with `.png` images.

The JSON file can store absolute paths or paths relative to the image folder. Only the final filename is used to load the image.

## `STL10UnsupervisedDataset`

Used for unlabeled data. Simply provide a directory with images. Filenames are sorted alphabetically to create a deterministic order.

## `STL10DINODataset`

Special dataset for DINO pretraining. It loads images from a folder like `STL10UnsupervisedDataset` but additionally generates multiple **global** and **local** views using provided transform functions. Parameters:

- `num_all_views` – total number of views to create.
- `num_global_views` – how many of those views should cover more than half of the image area.

---

## Directory expectations

- Images should be stored as `.png` files.
- Annotation files (`.csv` or `.json`) reside alongside or above the image folder.
- `root_dir` passed to the dataset points to the folder containing the images.

These simple structures keep the examples short and focused on the SSL algorithms rather than data loading.