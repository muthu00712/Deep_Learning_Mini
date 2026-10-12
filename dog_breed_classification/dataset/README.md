# Dataset Directory

This directory is the designated location for the Kaggle Dog Breed Identification dataset:
- `train/`: Directory containing training images (e.g. `000bec180eb18c7604dcecc8fe0dba07.jpg`, etc.)
- `labels.csv`: CSV file mapping each image ID to its dog breed (`id,breed`)

## Configurable Dataset Path
To avoid duplicating the large image dataset:
- The analysis script automatically checks relative paths (`./dataset`, `../train`, `../../train`, etc.).
- You can specify a custom dataset location using command-line arguments:
  ```bash
  python src/dataset_analysis.py --data-dir ../train --labels ../labels.csv
  ```
- Or via environment variables:
  ```bash
  export DATASET_DIR=../train
  export LABELS_CSV=../labels.csv
  ```
