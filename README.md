# Skin Disease Classification using CNN

## 📌 Project Overview
This project focuses on classifying different types of skin diseases using
Convolutional Neural Networks (CNN). The system is trained on dermoscopic
images from the HAM10000 dataset and aims to support AI-assisted early
detection of skin conditions.

The project includes dataset analysis, preprocessing, model training,
evaluation, and optional web-based prediction interface.

---

## 🎯 Objectives
- Analyze and prepare the HAM10000 dataset
- Handle class imbalance
- Perform stratified dataset splitting
- Train CNN-based classification model
- Evaluate model performance using multiple metrics
- Support reproducible AI healthcare research

---

## 📂 Dataset Structure

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

---

## 📊 Dataset Information

### HAM10000 Dataset
The dataset contains dermoscopic images of pigmented skin lesions collected
from multiple populations and imaging sources.

- Total Images: **10,015**
- Classes: **7 Skin Diseases**
- Image Format: RGB JPG

### Disease Classes

| Label | Disease Name |
|---------|------------------------------|
| nv | Melanocytic nevi |
| mel | Melanoma |
| bkl | Benign keratosis-like lesions |
| bcc | Basal cell carcinoma |
| akiec | Actinic keratoses |
| vasc | Vascular lesions |
| df | Dermatofibroma |

---

## 📥 Dataset Download

Due to GitHub file size limitations, the HAM10000 image dataset is not stored
directly in this repository.

Download dataset from:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

After downloading, extract dataset into:
data/raw/HAM10000_images/

---

## ⚙️ Dataset Preparation

### Class Distribution Analysis
- Identified severe class imbalance
- Visualization available in:
plots/class_distribution.png

### Class Imbalance Handling
Class weights were computed using a balanced weighting strategy to improve
model learning across minority disease classes.

### Dataset Splitting
Stratified split was applied:

| Split | Images |
|----------|------------|
| Training | 8012 |
| Validation | 1001 |
| Testing | 1002 |

Metadata split files are stored in:
processed/

---

## 🧪 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV
- TensorFlow / Keras (for CNN model training)

---

## ▶️ How to Run Scripts

### Install Dependencies
pip install pandas numpy matplotlib scikit-learn opencv-python

---

### Run Dataset Analysis
python scripts/load_data.py

---

### Compute Class Weights
python scripts/imbalance_handling.py

---

### Perform Dataset Split
python scripts/split_dataset.py

---

## 📈 Output Files

- Class distribution plot
- Stratified metadata splits
- Dataset analysis report

---

## 🧠 Key Features

- Efficient dataset organization
- Metadata-driven dataset splitting
- Class imbalance handling
- Reproducible ML pipeline
- Clean modular project design

---

## ⚠️ Ethical Considerations

- Dataset contains medical imagery
- No personal patient data is stored
- Images are used only for academic research purposes
- The system is intended to assist, not replace, medical professionals

---

## 🔮 Future Scope

- Implement advanced CNN architectures
- Add transfer learning models
- Integrate real-time prediction UI
- Extend dataset diversity
- Improve minority class accuracy

---
