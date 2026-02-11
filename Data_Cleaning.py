# ============================================================================
# 0. ライブラリのインポートと初期設定
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import os
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d')

# ============================================================================
# データの読み込みとデータ準備
# ============================================================================
DATA_DIR = '/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data'
DATA_DESCRIPTION_OUTPUT_DIR = f'/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data/data_description/'
# Ensure output directory exists
os.makedirs(DATA_DESCRIPTION_OUTPUT_DIR, exist_ok=True)

# Import data
ORIGINAL_DATA_NAME = 'analysisData_20260211'
ORIGINAL_DATA_PATH = f'{DATA_DIR}/{ORIGINAL_DATA_NAME}.csv'

data0 = pd.read_csv(ORIGINAL_DATA_PATH)
print(f"Successfully loaded data from {ORIGINAL_DATA_PATH}")
print(f"Shape: {data0.shape}")

# Save columns to text file
columns_output_path = f'{DATA_DESCRIPTION_OUTPUT_DIR}/{ORIGINAL_DATA_NAME}_colnames.txt'
with open(columns_output_path, 'w') as f:
    for col in data0.columns:
        f.write(f"{col}\n")
print(f"Columns names saved to {columns_output_path}")

# Convert date to datetime
data0['date'] = pd.to_datetime(data0['date'])

"""# Save data description to CSV
description_output_path = f'{DATA_DESCRIPTION_OUTPUT_DIR}/{ORIGINAL_DATA_NAME}_description.csv'
data0.describe().to_csv(description_output_path)
print(f"Data description saved to {description_output_path}")"""

# ============================================================================
# 変数のMAPPING
# ============================================================================

# Define the mappings
abuse_map = {
    1: "Physical Abuse",
    2: "Neglect",
    3: "Emotional Abuse",
    4: "Sexual Abuse",
    5: "Delinquency",
    6: "Parenting Difficulties",
    7: "Others"
}

occlusal_map = {
    1: "Normal Occlusion",
    2: "Crowding",
    3: "Anterior Crossbite",
    4: "Open Bite",
    5: "Maxillary Protrusion",
    6: "Crossbite",
    7: "Others"
}

need_treated_map = {
    1: "No Treatment Required",
    2: "Treatment Required"
}

emergency_map = {
    1: "Urgent Treatment Required"
}

gingivitis_map = {
    1: "No Gingivitis",
    2: "Gingivitis"
}

oral_clean_map = {
    1: "Poor",
    2: "Fair",
    3: "Good"
}

habits_map = {
    1: "None",
    2: "Digit Sucking",
    3: "Nail biting",
    4: "Tongue Thrusting",
    5: "Smoking",
    6: "Others"
}

# Apply the mappings
# Using .replace() ensures that any values not listed (like NaNs) remain unchanged
data0['abuse'] = data0['abuse'].replace(abuse_map)
data0['occlusalRelationship'] = data0['occlusalRelationship'].replace(occlusal_map)
data0['needTOBEtreated'] = data0['needTOBEtreated'].replace(need_treated_map)
data0['emergency'] = data0['emergency'].replace(emergency_map)
data0['gingivitis'] = data0['gingivitis'].replace(gingivitis_map)
data0['OralCleanStatus'] = data0['OralCleanStatus'].replace(oral_clean_map)
data0['habits'] = data0['habits'].replace(habits_map)

# Verify the changes
cols_to_check = ['abuse', 'occlusalRelationship', 'needTOBEtreated', 'emergency', 'gingivitis', 'OralCleanStatus', 'habits']
# Check if any numeric values remain
print("\n--- Mapping Check ---")
for col in cols_to_check:
    # Get unique values, dropping NaNs
    unique_vals = data0[col].dropna().unique()
    
    # Check if any value is a number (int or float, excluding boolean)
    # Note: numpy types like np.int64 are also numbers
    remaining_nums = [x for x in unique_vals if isinstance(x, (int, float, np.number)) and not isinstance(x, bool)]
    
    if len(remaining_nums) > 0:
        print(f"{col}: Remain Num (Values: {remaining_nums})")
    else:
        print(f"{col}: OK")

# ============================================================================
# Set the logical order
# ============================================================================
# 1. Define the logical order for each column
abuse_order = [
    "Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse", 
    "Delinquency", "Parenting Difficulties", "Others"
]
occlusal_order = [
    "Normal Occlusion", "Crowding", "Anterior Crossbite", "Open Bite", 
    "Maxillary Protrusion", "Crossbite", "Others"
]
need_treated_order = ["No Treatment Required", "Treatment Required"]
emergency_order = ["Urgent Treatment Required"]
gingivitis_order = ["No Gingivitis", "Gingivitis"]
oral_clean_order = ["Poor", "Fair", "Good"]
habits_order = [
    "None", "Digit Sucking", "Nail biting", "Tongue Thrusting", 
    "Smoking", "Others"
]

# 2. Convert columns to Categorical with the specified order
data0['abuse'] = pd.Categorical(data0['abuse'], categories=abuse_order, ordered=True)
data0['occlusalRelationship'] = pd.Categorical(data0['occlusalRelationship'], categories=occlusal_order, ordered=True)
data0['needTOBEtreated'] = pd.Categorical(data0['needTOBEtreated'], categories=need_treated_order, ordered=True)
data0['emergency'] = pd.Categorical(data0['emergency'], categories=emergency_order, ordered=True)
data0['gingivitis'] = pd.Categorical(data0['gingivitis'], categories=gingivitis_order, ordered=True)
data0['OralCleanStatus'] = pd.Categorical(data0['OralCleanStatus'], categories=oral_clean_order, ordered=True)
data0['habits'] = pd.Categorical(data0['habits'], categories=habits_order, ordered=True)

print("Columns converted to ordered categories.")

# Save cleaned data to CSV
cleaned_data_output_path = f'{DATA_DIR}/{ORIGINAL_DATA_NAME}_cleaned.csv'
data0.to_csv(cleaned_data_output_path, index=False)
print(f"Cleaned data saved to {cleaned_data_output_path}")
