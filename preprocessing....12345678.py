import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image

INPUT_FOLDER = "dataset"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not os.path.exists(INPUT_FOLDER):
    print("Dataset folder missing!")
    exit()

hashes = set()
report = []
brightness_groups = {"dark":0,"medium":0,"bright":0}

# ===== IMAGE QUALITY =====
def image_quality_score(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray,cv2.CV_64F).var()

# ===== SMART NOISE REDUCTION =====
def smart_noise_reduction(img):
    std=np.std(img)
    if std>50:
        return cv2.GaussianBlur(img,(5,5),0),"Gaussian"
    else:
        return cv2.medianBlur(img,5),"Median"

print("Processing Started...\n")

for img_name in os.listdir(INPUT_FOLDER):

    path=os.path.join(INPUT_FOLDER,img_name)
    img=cv2.imread(path)

    if img is None:
        continue

    print("Processing:",img_name)

    # Resize
    resized=cv2.resize(img,(224,224))

    # Normalize
    normalized=resized/255.0

    # Quality score
    quality=image_quality_score(resized)

    # Duplicate check
    duplicate="No"
    h=hash(resized.tobytes())
    if h in hashes:
        duplicate="Yes"
    hashes.add(h)

    # Noise reduction
    denoise,noise_type=smart_noise_reduction(resized)

    # Resolution
    pil_img=Image.open(path)
    resolution=pil_img.size

    # Brightness analysis
    brightness=np.mean(resized)

    if brightness<80:
        group="dark"
    elif brightness<160:
        group="medium"
    else:
        group="bright"

    brightness_groups[group]+=1

    # Save processed image
    cv2.imwrite(os.path.join(OUTPUT_FOLDER,"processed_"+img_name),denoise)

    # Add to CSV report
    report.append([
        img_name,
        quality,
        brightness,
        resolution,
        duplicate,
        noise_type,
        group,
        "Processed"
    ])

# ===== DATASET IMBALANCE CHECK =====
imbalance="No"
if max(brightness_groups.values())>2*min(brightness_groups.values()):
    imbalance="Yes"

# Add imbalance info
for row in report:
    row.append(imbalance)

# ===== SAVE CSV =====
df=pd.DataFrame(report,columns=[
    "Image",
    "QualityScore",
    "Brightness",
    "Resolution",
    "Duplicate",
    "NoiseFilterUsed",
    "BrightnessGroup",
    "Status",
    "DatasetImbalance"
])

df.to_csv("dataset_report.csv",index=False)

print("\nProcessing Completed Successfully!")
print("CSV Report Saved → dataset_report.csv")
