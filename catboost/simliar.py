import pandas as pd

# Load your dataset
dataset_path = r"C:\Users\VICTUS\Desktop\XX_XX\Lightgbm\lightgbm dataset.csv"
df = pd.read_csv(dataset_path)

# Drop rows with missing values (optional)
df.dropna(inplace=True)

# Find duplicate rows
duplicates = df[df.duplicated()]

# Count duplicate rows
num_duplicates = len(duplicates)

# Print result
print(f"Number of similar (duplicate) rows: {num_duplicates}")
