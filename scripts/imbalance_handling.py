import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Load metadata
CSV_PATH = "data/raw/HAM10000_metadata.csv"
df = pd.read_csv(CSV_PATH)

# Unique classes
classes = df['dx'].unique()

# Compute class weights
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=df['dx']
)

class_weights = dict(zip(classes, weights))

print("Class Weights:\n")
for cls, wt in class_weights.items():
    print(f"{cls}: {wt:.4f}")
