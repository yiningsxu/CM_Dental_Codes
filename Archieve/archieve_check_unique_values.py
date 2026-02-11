"""
・アーカイブ入り・
最新コードにfunction化。

Created: 2025-11-29
Last Edited: 2026-02-11
Summary: Loads 'data_OnlyAbuse_N1235.csv', calculates unique value counts for each column (excluding specific ones), and saves the result to 'unique_values_summary.csv'.
"""
import pandas as pd
import sys
import os 
import datetime

timestamp = datetime.datetime.now().strftime('%Y%m%d')

# Load the dataset
# ============================================================================
INPUT_DIR = '/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data'
OUTPUT_DIR = f'/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/result/{timestamp}/'
csv_name = "data_OnlyAbuse_N1235.csv"
csv_path = f"{INPUT_DIR}/{csv_name}"
output_path = f"{OUTPUT_DIR}/unique_values_summary_{csv_name}.csv"


try:
    data0 = pd.read_csv(csv_path)
    print(f"Successfully loaded data from {csv_path}")
    print(f"Shape: {data0.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {csv_path}")
    sys.exit(1)

# Collect unique values and counts for each column
results = []

for col in data0.columns:
    if col in ["Unnamed: 0","No_All","instruction_detail", "instruction", "memo"]:
        continue
        
    value_counts = data0[col].value_counts(dropna=False)
    
    for val, count in value_counts.items():
        results.append({
            "Column": col,
            "Value": val,
            "Count": count
        })

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)

print(f"Summary saved to {output_path}")
