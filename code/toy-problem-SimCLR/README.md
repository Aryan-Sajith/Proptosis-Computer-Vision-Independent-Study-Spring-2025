For the other toy-problem we only used rotation prediction and predicted between two different distinct face types(which doesn't necessarily best model how proptosis detection would work with SSL), we've decided to pick a different task with a different dataset:

# Proptosis SSL Toy Project - SimCLR

## Overview

This folder implements a toy-phase experiment to demonstrate the benefits of self-supervised learning (SSL) with SimCLR in low-data regimes. We use the UTKFace dataset to pre-train a ResNet-based encoder and then fine-tune it on a small age‐group classification task (<20 vs. >60 years). A from-scratch CNN baseline is compared against the SSL‐pretrained model under identical few-shot conditions.

## Directory Structure

```
proptosis-ssl-toy/
├── data/                  # dataset here
│   └── UTKFace/           # The raw images are here
├── splits/
│   └── fewshot_splits.json  # Generated train/val/test indices
├── src/                   # Source code
│   ├── datasets.py        # UTKFace loader & split logic
│   ├── models.py          # Encoder, projection head, classifier definitions
│   ├── utils.py           # Loss functions, metrics, training loops
│   ├── train_simclr.py    # SSL pre-training script
│   └── train_supervised.py # Supervised / SSL fine-tune script
├── configs/               # YAML configs for experiments
│   ├── simclr.yaml
│   └── supervised.yaml
├── checkpoints/           # Saved model weights
├── logs/                  # TensorBoard or wandb logs
└── README.md              # This file
```

## Prerequisites

* Python 3.8+
* PyTorch 1.10+ and torchvision
* yaml, numpy, pillow
* Optionally: TensorBoard or Weights & Biases for logging

Install dependencies:

```bash
pip install torch torchvision pyyaml numpy pillow
# For logging: pip install tensorboard wandb
```

## Dataset Preparation

1. Download the [UTKFace dataset](https://susanqq.github.io/UTKFace/) and extract it under `data/UTKFace/`.
2. Ensure images are named in the format `[age]_[gender]_[race]_[timestamp].jpg`.

## Generating Few-Shot Splits

Use the provided script to sample 100 images per age-bin and split them:

```bash
python - << 'EOF'
from src.datasets import make_fewshot_splits
make_fewshot_splits(
    root_dir='data/UTKFace',
    out_file='splits/fewshot_splits.json',
    seed=42,
    k_per_class=100,
    age_bins=(20,60),
    val_per_class=10,
    test_per_class=10
)
EOF
```

This creates `splits/fewshot_splits.json` with `train`, `val`, and `test` lists.

## SSL Pre-Training (SimCLR)

Train the encoder on the unlabeled portion of UTKFace:

```bash
python src/train_simclr.py --cfg configs/simclr.yaml
```

* **Output**: checkpoints/simclr\_ep\*.pt
* **Config**: batch size, learning rate, epochs, temperature, augmentations in `configs/simclr.yaml`

## Supervised vs. SSL Fine-Tuning

1. **From-scratch baseline**:

   ```bash
   python src/train_supervised.py --cfg configs/supervised.yaml --mode scratch
   ```
2. **SSL fine-tune**:

   ```bash
   python src/train_supervised.py --cfg configs/supervised.yaml --mode ssl_ft
   ```

* Both use the same 80/10 train/val splits defined in `splits/fewshot_splits.json`.
* Model weights are saved in `checkpoints/{mode}_best.pt`.

## Evaluation & Reporting

* Evaluate on the held-out test set (10 images/class) using the best validation checkpoint.
* Metrics: accuracy, ROC-AUC, precision, recall.
* Repeat the entire process with 3–5 random seeds and report mean ± std.
* Optional: k-NN probe on frozen embeddings to assess separability.

## Visualization & Logging

* Track training/validation loss and accuracy curves with TensorBoard or wandb.
* Plot performance vs. shot count (*k*) and compare scratch vs. SSL.
* Use t-SNE/UMAP to visualize embedding spaces before and after fine-tuning.

## Configuration Parameters

* **configs/simclr.yaml**: data paths, augmentations, batch size, learning rate, epochs, temperature.
* **configs/supervised.yaml**: data paths, which checkpoint to load (`pretrained_ckpt`), training hyperparameters.

## Citation

SimCLR Paper can be found here: https://arxiv.org/abs/2002.05709
