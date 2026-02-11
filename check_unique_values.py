import pandas as pd

# Load the dataset
csv_path = "/Users/yining/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data/data_original_analysis_final_20251113.csv"
try:
    data0 = pd.read_csv(csv_path)
    print(f"Successfully loaded data from {csv_path}")
    print(f"Shape: {data0.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}")
    exit(1)

# Print unique values for each column
print("\n--- Unique Values per Column ---\n")
for col in data0.columns:
    unique_vals = data0[col].unique()
    print(f"--- {col} (Count: {len(unique_vals)}) ---")
    print(unique_vals)
    print("\n")
