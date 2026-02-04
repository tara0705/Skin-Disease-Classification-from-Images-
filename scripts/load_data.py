import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
CSV_PATH = "data/raw/HAM10000_metadata.csv"
PLOT_DIR = "plots"

# Create plots folder if not exists
os.makedirs(PLOT_DIR, exist_ok=True)

# Load metadata
df = pd.read_csv(CSV_PATH)

# Class distribution
class_counts = df['dx'].value_counts()
print("Class Distribution:\n")
print(class_counts)

# Plot class distribution
plt.figure(figsize=(8, 5))
class_counts.plot(kind='bar', color='steelblue')
plt.title("Class Distribution of HAM10000 Dataset")
plt.xlabel("Skin Disease Class")
plt.ylabel("Number of Images")
plt.tight_layout()

# Save plot
plt.savefig(os.path.join(PLOT_DIR, "class_distribution.png"))
plt.show()
