# Galaxy Zoo — CNN-based Galaxy Classification and Regression

This repository contains the code and numerical notebooks developed for the course Physics Applications of AI of the Master in Cosmology and Astroparticles Physics, University of Geneva. This project applies Convolutional Neural Networks (CNNs) to the [Galaxy Zoo](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge) dataset, in which citizen scientists classified galaxy morphology through an online survey. The goal is to predict both discrete morphological classes and continuous vote fractions from galaxy images, replicating and extending the original Kaggle challenge.

The dataset consists of ~61,000 SDSS galaxy images (424×424 px, RGB) paired with crowd-sourced vote fractions across a hierarchical decision tree of morphological questions.


## Project Structure

```
galaxyzoo/
├── data/
│   ├── images/          # Galaxy images (.jpg)
│   └── labels.csv       # Vote fractions per galaxy
├── src/galaxyzoo/
│   ├── cnn.py           # CNN architecture
│   ├── data.py          # Data loading and dataset classes
│   └── training_utils.py
├── task1.1.ipynb        # 3-class classification
├── task1.2.ipynb        # Binary classification
├── task2.ipynb          # Regression (5 outputs)
└── task3.ipynb          # Regression (14 outputs)
```


## Modules

### `src/galaxyzoo/cnn.py`

Defines the CNN architecture used across all tasks.

- **`DoubleConvolutionBlock`**: two consecutive Conv2d layers, each followed by BatchNorm and ReLU. Supports residual connections — when `in_channels != out_channels`, a 1×1 convolution projects the residual to the right shape.
- **`CNN`**: stacks `n_blocks` (default 3) `DoubleConvolutionBlock` + `MaxPool2d(2,2)` layers to progressively halve the spatial resolution and double the number of channels. After the convolutional blocks, an `AdaptiveAvgPool2d(1,1)` collapses the spatial dimensions to a single value per channel, making the model resolution-agnostic, followed by a two-layer MLP head. For regression tasks a final `Sigmoid` is added to bound outputs to [0, 1].

### `src/galaxyzoo/data.py`

- **`load_data`**: reads `labels.csv` and returns image paths with the corresponding labels for a given task (`task1`, `task2`, `task3`). For `task1` the vote fractions are converted to hard class labels via argmax; a `binary` flag further filters to high-confidence smooth/disk examples (>75% vote share, artifacts excluded).
- **`GalaxiesDataset`**: a PyTorch `Dataset` that loads images on-the-fly via PIL and applies the provided transforms. Returns `torch.long` labels for classification tasks and `torch.float32` for regression.

### `src/galaxyzoo/training_utils.py`

- **`perform_train_loop`** / **`perform_validation_loop`**: single-epoch loops that set `model.train()` / `model.eval()` appropriately (critical for BatchNorm behaviour). Binary classification predictions are obtained via `.flatten()` to match the 1D label shape expected by `BCEWithLogitsLoss`.
- **`fit`**: full training loop over `epochs`, calling both loops, stepping the scheduler on validation loss, and optionally logging accuracy.


## Tasks

### Task 1.1 — 3-class Classification

Classifies each galaxy into one of three morphological categories: **Smooth**, **Disk**, or **Artifact**, based on the Q1 vote fractions (`Class1.1`, `Class1.2`, `Class1.3`).

- Loss: `CrossEntropyLoss` with class weights computed from the training set to handle class imbalance.
- Evaluation: confusion matrix, ROC curves (one-vs-rest), test accuracy with misclassified examples.
- Final result: ~85% training accuracy, ~85% test accuracy (25 epochs).

### Task 1.2 — Binary Classification

Restricts to galaxies where the dominant Q1 vote exceeds 75% confidence and the predicted class is either Smooth or Disk (artifacts excluded). Trains a binary classifier on this filtered subset.

- Loss: `BCEWithLogitsLoss`.
- Evaluation: confusion matrix, ROC curve.
- Final result: ~95% test accuracy (20 epochs).

### Task 2 — Regression (5 outputs)

Predicts five continuous vote fractions simultaneously: `Class2.1`, `Class2.2`, `Class7.1`, `Class7.2`, `Class7.3`.

- Loss: `MSELoss`.
- Evaluation: histogram of true vs. predicted distributions, hexbin true-vs-predicted plots, MAE per output.
- The `Q2.2` output (disk fraction) shows the highest MAE due to its near-uniform distribution across galaxies.

### Task 3 — Regression (14 outputs) with Hierarchical Constraints

Predicts 14 vote fractions spanning the Galaxy Zoo decision tree: Q2, Q6, Q7, Q8. The Galaxy Zoo survey is hierarchical, which means that Q7 answers are only shown to respondents who chose Q1.1 (smooth), so the sum of Q7 fractions should equal the Q1.1 fraction; similarly, Q8 answers come from Q6.1 respondents.

A custom loss function enforces these physical constraints:

```python
loss = MSE + λ * (constraint_Q7 + constraint_Q8)
```

where `constraint_Q7 = (sum(Q7 predictions) − Q1.1_true)²` and `constraint_Q8 = (sum(Q8 predictions) − Q6.1_true)²`. A value of `λ = 0.05` was found to balance constraint enforcement with raw MSE performance.


## Usage

### 1. Install dependencies

The project uses [uv](https://github.com/astral-sh/uv) for environment management:

```bash
cd galaxyzoo
uv sync
```

Or with pip:

```bash
pip install torch torchvision matplotlib numpy pandas kagglehub scikit-learn
```

### 2. Download the data

The data is already included in the repository under `data/images/` (galaxy images) and `data/labels.csv` (vote fractions). No additional download is required.

If the data directory is missing, the dataset can be obtained from Kaggle and placed in the same structure.

### 3. Run the notebooks

Open any of the task notebooks in JupyterLab or VS Code and run all cells in order. Each notebook is self-contained: it loads data, trains the model, and evaluates results.


## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.11 | Model definition, training |
| `torchvision` | ≥ 0.26 | Image transforms |
| `numpy` | ≥ 2.4 | Numerical operations |
| `pandas` | ≥ 3.0 | Label loading |
| `matplotlib` | ≥ 3.10 | Plots and visualisations |
| `scikit-learn` | — | Metrics, class weights, train/test split |
| `kagglehub` | ≥ 1.0 | Dataset download |
| `Pillow` | — | Image loading |

Python ≥ 3.14 required.

## Author

Developed by Jose Carlos Díaz Sierra for the Physics Applications of AI course, as part of Master in Cosmology and Astroparticles Physics at University of Geneva.