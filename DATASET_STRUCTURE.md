# Project Structure
## Skin Disease Classification using CNN

DATA_ANALYSIS/
├── data/
│ ├── raw/
│ │ ├── HAM10000_images/ (all 10,015 images)
│ │ └── HAM10000_metadata.csv
│ └── processed/
│ ├── train_metadata.csv
│ ├── val_metadata.csv
│ └── test_metadata.csv
├── notebooks/
│ └── eda.ipynb
├── plots/
│ └── class_distribution.png
├── scripts/
│ ├── load_data.py
│ ├── imbalance_handling.py
│ └── split_dataset.py
└── dataset_report.md

---

## 📦 External Dataset

Due to GitHub storage limitations, the HAM10000 image dataset is not stored directly in this repository.

Dataset Source:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

After downloading, extract dataset into:
Path: DATA_ANALYSIS/data/raw/HAM10000_images/-(all 10,015 images)
