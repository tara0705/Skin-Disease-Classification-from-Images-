import pandas as pd
from sklearn.model_selection import train_test_split

# Load metadata
CSV_PATH = "data/raw/HAM10000_metadata.csv"
df = pd.read_csv(CSV_PATH)

# Train (80%) + temp (20%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['dx'],
    random_state=42
)

# Validation (10%) + Test (10%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df['dx'],
    random_state=42
)

print("Dataset Split Sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# Save CSVs (optional but good)
train_df.to_csv("data/processed/train_metadata.csv", index=False)
val_df.to_csv("data/processed/val_metadata.csv", index=False)
test_df.to_csv("data/processed/test_metadata.csv", index=False)
