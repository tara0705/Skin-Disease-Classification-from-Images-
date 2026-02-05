# Dataset Analysis Report  
## Skin Disease Classification using CNN (HAM10000)

---

## 1. Introduction

This report describes the dataset preparation and analysis process for the
Skin Disease Classification project. The objective of this phase is to ensure
that the HAM10000 dataset is clean, well-understood, balanced, and properly
structured for training a Convolutional Neural Network (CNN) model for
multi-class skin disease classification.

---

## 2. Dataset Description

The HAM10000 (Human Against Machine with 10000 training images) dataset is a
publicly available medical imaging dataset consisting of dermoscopic images of
common pigmented skin lesions.

- Total images: **10,015**
- Number of classes: **7**
- Image format: **RGB (.jpg)**
- Data source: **Kaggle / ISIC Archive**

### Disease Classes

| Label | Disease Name |
|------|-------------|
| nv   | Melanocytic nevi |
| mel  | Melanoma |
| bkl  | Benign keratosis-like lesions |
| bcc  | Basal cell carcinoma |
| akiec| Actinic keratoses |
| vasc | Vascular lesions |
| df   | Dermatofibroma |

The dataset also includes a metadata CSV file containing image IDs, diagnosis
labels, and patient-related information such as age, sex, and lesion
localization.

---

## 3. Dataset Collection and Organization

All dermoscopic images from the dataset were consolidated into a single
directory named `HAM10000_images`, while the corresponding metadata and class
labels were stored separately in `HAM10000_metadata.csv`.

Dataset integrity was verified by programmatically matching the number of image
files in the directory with the number of entries in the metadata file, ensuring
a one-to-one correspondence between images and labels.

This separation of raw images and metadata ensures clarity, reproducibility, and
scalability for further processing stages.

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Class Distribution Analysis

Exploratory data analysis was performed to understand the distribution of images
across different disease classes. The analysis revealed a significant class
imbalance in the dataset.

#### Class-wise Image Distribution

- nv: 6705 images  
- mel: 1113 images  
- bkl: 1099 images  
- bcc: 514 images  
- akiec: 327 images  
- vasc: 142 images  
- df: 115 images  

A bar chart visualization was generated to clearly illustrate this imbalance.

### Observation

The melanocytic nevi (nv) class dominates the dataset, while classes such as
dermatofibroma (df) and vascular lesions (vasc) are severely under-represented.
This imbalance can negatively impact model performance if not addressed
properly.

---

## 5. Class Imbalance Handling

To mitigate the effects of class imbalance, **class weighting** was adopted as
the primary strategy. Class weights were computed using a balanced weighting
scheme, assigning higher importance to minority classes and lower importance to
majority classes during model training.

This approach was preferred over oversampling techniques to avoid image
duplication and reduce the risk of overfitting, which is particularly important
in medical imaging tasks.

The computed class weights ensure that the CNN model learns meaningful features
from all disease categories rather than being biased toward the majority class.

---

## 6. Dataset Splitting Strategy

The dataset was split using a **stratified 80–10–10 strategy**, ensuring that
class proportions were preserved across all splits:

- Training set: **80% (8012 images)**
- Validation set: **10% (1001 images)**
- Test set: **10% (1002 images)**

Stratified splitting prevents data leakage and ensures reliable performance
evaluation by maintaining consistent class distributions across training,
validation, and testing phases.

Separate metadata files were generated for each split to maintain traceability
and reproducibility.

---

## 7. Final Dataset Structure

After analysis and preparation, the dataset follows a clean and modular
structure consisting of raw data, processed metadata splits, analysis scripts,
visualizations, and documentation. This structure is optimized for seamless
integration into CNN training pipelines.

---

## 8. Conclusion

This dataset analysis phase successfully prepared the HAM10000 dataset for
deep learning-based skin disease classification. Through careful exploration,
visualization, imbalance handling, and stratified splitting, the dataset is now
well-suited for training robust and unbiased CNN models.

Proper dataset preparation plays a critical role in medical AI applications, as
it directly impacts model accuracy, fairness, and generalization, especially in
high-stakes domains such as healthcare.

---

## 9. Future Scope

Future improvements may include:
- Advanced data augmentation for minority classes
- Cross-dataset validation using additional skin lesion datasets
- Bias analysis based on patient demographics
- Integration of metadata features alongside image features

---

**End of Report**
