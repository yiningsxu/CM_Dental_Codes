# ============================================================================
# 0. ライブラリのインポートと初期設定
# ============================================================================
print("\n------- Import Libraries ------- ")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import os
from datetime import datetime
from Functions import (
    save_value_counts_summary, 
    create_table1_demographics,
    create_table2_oral_health_descriptive,
    create_table3_statistical_comparisons,
    create_table4_multivariate_analysis,
    create_forest_plot_vertical,
    create_table5_dmft_by_lifestage_abuse,
    create_table5_5_caries_prevalence_treatment,
    create_table6_dmft_by_dentition_abuse,
    analyze_dmft_by_dentition_with_pairwise,
    create_visualizations,
    plot_boxplot_with_dunn,
    plot_boxplot_by_dentition_type,
    generate_summary_report,
    simple_logistic_regression
)

timestamp = datetime.now().strftime('%Y%m%d')

# ============================================================================
# データの読み込みとデータ準備
# ============================================================================
print("==================== All Available Data (until 2025/02) ====================")
print("\n------- Load Data ------- ")
DATA_DIR = '/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data'
DATA_DESCRIPTION_OUTPUT_DIR = f'/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data/data_description/'
# Ensure output directory exists
os.makedirs(DATA_DESCRIPTION_OUTPUT_DIR, exist_ok=True)

# Define OUTPUT_DIR for main analysis results
OUTPUT_DIR = f'/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/result/{timestamp}/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

print("\n------- Convert Date to Datetime ------- ")
# Convert date to datetime
data0['date'] = pd.to_datetime(data0['date'])

# ============================================================================
# 変数のMAPPING
# ============================================================================
print("\n------- Variable Mapping ------- ")
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
print("\n------- Set Variable Order ------- ")
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

print("\n--- Save Mapped & Ordered Data ---")
# Save cleaned data to CSV
cleaned_data_output_path = f'{DATA_DIR}/{ORIGINAL_DATA_NAME}_AllData_cleaned.csv'
data0.to_csv(cleaned_data_output_path, index=False)
print(f"Cleaned data saved to {cleaned_data_output_path}")

# ============================================================================
# Data Summary
# ============================================================================
print("\n------- Data Summary ------- ")
# Counts for every variable
output_path = f'{DATA_DESCRIPTION_OUTPUT_DIR}/unique_values_summary_{ORIGINAL_DATA_NAME}.csv'
exclude_cols = ["No_All", "instruction_detail", "instruction", "memo"]
save_value_counts_summary(data0, output_path, exclude_cols)

# Data description
# Summary of statistics (count, mean, std, min, 25%, 50%, 75%, max)
description_output_path = f'{DATA_DESCRIPTION_OUTPUT_DIR}/{ORIGINAL_DATA_NAME}_description.csv'
data0.describe().to_csv(description_output_path)
print(f"Data description saved to {description_output_path}")

# ============================================================================
# Filter data for dates on or before March 31, 2024
# ============================================================================
print("==================== Analysis Data (until 2024/03) ====================")
print("\n------- Extract data <= March 31, 2024 ------- ")
data_untilMar2024 = data0[data0['date'] <= '2024-03-31']

# Check the results
print(f"Original row count: {len(data0)}")
print(f"Filtered row count: {len(data_untilMar2024)}")
print(f"Date range in filtered data: {data_untilMar2024['date'].min()} to {data_untilMar2024['date'].max()}")

# Save the filtered data to a new CSV file
csv_name = f"{ORIGINAL_DATA_NAME}_tillMar2024"
output_path = f'{DATA_DIR}/{csv_name}.csv'
data_untilMar2024.to_csv(output_path, index=False)

# Exclude data with NA in Abuse
# Keep only rows where abuse_num is NOT 0
print("\n------- Exclude Abuse Num NA ------- ")
data_untilMar2024_noAbuseNA = data_untilMar2024[data_untilMar2024['abuse_num'] != 0]

# Verify the result
print(f"Original row count: {len(data_untilMar2024)}")
print(f"No Abuse Num NA rows: {len(data_untilMar2024_noAbuseNA)}")

# Only abuse in 1-4
print("\n------- Only Abuse in 1-4 & Abuse Num is 1 ------- ")
# Filter data based on the two conditions
# 1. abuse_num is 1
# 2. abuse is between 1 and 4 (inclusive)
target_abuse_types = [
    "Physical Abuse", 
    "Neglect", 
    "Emotional Abuse", 
    "Sexual Abuse"
]

data_noNA_Only1 = data_untilMar2024_noAbuseNA[
    (data_untilMar2024_noAbuseNA['abuse_num'] == 1) & 
    (data_untilMar2024_noAbuseNA['abuse'].isin(target_abuse_types))
]

# Verify the results
print(f"No Abuse Num NA rows: {len(data_untilMar2024_noAbuseNA)}")
print(f"Filtered row count: {len(data_noNA_Only1)}")
print(f"Only single abuse 1-4: {data_noNA_Only1['abuse'].unique()}")

print("\n------- Save Final Data for Analysis ------- ") 
# Save the filtered data to a new CSV file
csv_name = f"{ORIGINAL_DATA_NAME}_tillMar2024_noAbuseNA_Only1"
output_path = f'{DATA_DIR}/{csv_name}.csv'
data_noNA_Only1.to_csv(output_path, index=False)

# Counts for every variable
output_path = f'{DATA_DESCRIPTION_OUTPUT_DIR}/unique_values_summary_{csv_name}.csv'
exclude_cols = ["No_All", "instruction_detail", "instruction", "memo"]
save_value_counts_summary(data_noNA_Only1, output_path, exclude_cols)

"""
"""
"""
"""

print("=" * 70)
print("虐待分類と口腔内状況の関連に関する解析")
print("Analysis of Oral Health Conditions by Child Abuse Type")
print("=" * 70)
print()

# Use the already prepared dataframe
df = data_noNA_Only1
output_dir = OUTPUT_DIR

# Add age_group
print("\n------- Calculating Derived Variables ------- ")
df['age_group'] = pd.cut(df['age_year'], 
                         bins=[0, 6, 12, 18],
                         labels=['Early Childhood (2-6)', 
                                'Middle Childhood (7-12)', 
                                'Adolescence (13-18)'],
                         right=True)

# Define tooth columns
perm_teeth_cols = [f'U{i}{j}' for i in [1, 2] for j in range(1, 8)] + \
                  [f'L{i}{j}' for i in [3, 4] for j in range(1, 8)]
baby_teeth_cols = [f'u{i}{j}' for i in [5, 6] for j in range(1, 6)] + \
                  [f'l{i}{j}' for i in [7, 8] for j in range(1, 6)]

# Ensure columns exist
perm_teeth_cols = [c for c in perm_teeth_cols if c in df.columns]
baby_teeth_cols = [c for c in baby_teeth_cols if c in df.columns]

print(f"   Found {len(perm_teeth_cols)} permanent teeth columns and {len(baby_teeth_cols)} baby teeth columns.")

# Calculate Permanent components
# 2: Decayed, 3: Filled, 4: Missing (Assumed mapping)
# 0: Sound
df['Perm_D'] = df[perm_teeth_cols].apply(lambda x: (x == 2).sum(), axis=1)
df['Perm_M'] = df[perm_teeth_cols].apply(lambda x: (x == 4).sum(), axis=1)
df['Perm_F'] = df[perm_teeth_cols].apply(lambda x: (x == 3).sum(), axis=1)
df['Perm_Sound'] = df[perm_teeth_cols].apply(lambda x: (x == 0).sum(), axis=1)
df['Perm_DMFT'] = df['Perm_D'] + df['Perm_M'] + df['Perm_F']

# Calculate Baby components
df['Baby_d'] = df[baby_teeth_cols].apply(lambda x: (x == 2).sum(), axis=1)
df['Baby_m'] = df[baby_teeth_cols].apply(lambda x: (x == 4).sum(), axis=1) # Note: Missing baby teeth often not recorded as 4
df['Baby_f'] = df[baby_teeth_cols].apply(lambda x: (x == 3).sum(), axis=1)
df['Baby_sound'] = df[baby_teeth_cols].apply(lambda x: (x == 0).sum(), axis=1)
df['Baby_DMFT'] = df['Baby_d'] + df['Baby_m'] + df['Baby_f']

# Total DMFT
df['DMFT_Index'] = df['Perm_DMFT'] + df['Baby_DMFT']

# Calculate Care Index (Filled / DMFT * 100)
# Handle division by zero: if DMFT=0, Care Index is NaN
df['Care_Index'] = (df['Perm_F'] + df['Baby_f']) / df['DMFT_Index'] * 100
df['Care_Index'] = df['Care_Index'].replace([np.inf, -np.inf], np.nan)

# Calculate Healthy Rate (Sound / (Sound + DMFT) * 100)
# Denominator = Sound + D + M + F (Total examined teeth)
total_teeth = df['Perm_Sound'] + df['Baby_sound'] + df['DMFT_Index']
df['Healthy_Rate'] = (df['Perm_Sound'] + df['Baby_sound']) / total_teeth * 100
df['Healthy_Rate'] = df['Healthy_Rate'].replace([np.inf, -np.inf], np.nan)

# Add caries indicators
df['has_caries'] = (df['DMFT_Index'] > 0).astype(int)
df['has_untreated_caries'] = ((df['Perm_D'] + df['Baby_d']) > 0).astype(int)

print(f"   Calculated DMFT_Index. Mean: {df['DMFT_Index'].mean():.2f}")
print(f"   Calculated Care_Index. Mean: {df['Care_Index'].mean():.2f}")
print(f"   Calculated Healthy_Rate. Mean: {df['Healthy_Rate'].mean():.2f}")

print(f"   Total samples: {len(df)}")
print(f"   Abuse types: {df['abuse'].value_counts().to_dict()}")
print()

print("2. Creating Table 1: Demographic Characteristics...")
table1 = create_table1_demographics(df)
table1.to_csv(f'{output_dir}table1_demographics_{timestamp}.csv', index=False)
print(f"   ✓ Saved: table1_demographics_{timestamp}.csv")
print()

print("3. Creating Table 2: Oral Health Descriptive Statistics...")
table2_cont, table2_cat = create_table2_oral_health_descriptive(df)
table2_cont.to_csv(f'{output_dir}table2_continuous_{timestamp}.csv', index=False)
table2_cat.to_csv(f'{output_dir}table2_categorical_{timestamp}.csv', index=False)
print(f"   ✓ Saved: table2_continuous_{timestamp}.csv")
print(f"   ✓ Saved: table2_categorical_{timestamp}.csv")
print()

print("4. Creating Table 3: Statistical Comparisons...")
table3_overall, table3_posthoc, table3_pairwise, table3_tidy_posthoc = create_table3_statistical_comparisons(df)
table3_overall.to_csv(f'{output_dir}table3_overall_tests_{timestamp}.csv', index=False)
table3_posthoc.to_csv(f'{output_dir}table3_posthoc_{timestamp}.csv', index=False)
table3_pairwise.to_csv(f'{output_dir}table3_pairwise_{timestamp}.csv', index=False)

print(f"   ✓ Saved: table3_overall_tests_{timestamp}.csv")
print(f"   ✓ Saved: table3_posthoc_{timestamp}.csv")
print(f"   ✓ Saved: table3_pairwise_{timestamp}.csv")
print()

print("5. Creating Table 4: Multivariate Analysis...")
table4 = create_table4_multivariate_analysis(df)
table4.to_csv(f'{output_dir}table4_logistic_regression_{timestamp}.csv', index=False)
print(f"   ✓ Saved: table4_logistic_regression_{timestamp}.csv")
print()

print("5.1 Creating Forest Plot for Logistic Regression Results...")
create_forest_plot_vertical(table4, df, output_dir, timestamp)
print()

print("5.2 Creating Table 5: DMFT by Life Stage and Abuse Type...")
table5, table5_tidy_posthoc = create_table5_dmft_by_lifestage_abuse(df)
if not table5.empty:
    table5.to_csv(f'{output_dir}table5_dmft_lifestage_abuse_{timestamp}.csv', index=False)
    print(f"   ✓ Saved: table5_dmft_lifestage_abuse_{timestamp}.csv")
print()

print("5.2.5 Creating Table 5.5: Caries Prevalence and Treatment Status...")
table5_5, table5_5_tidy_posthoc = create_table5_5_caries_prevalence_treatment(df)
if not table5_5.empty:
    table5_5.to_csv(f'{output_dir}table5_5_caries_prevalence_treatment_{timestamp}.csv', index=False)
    print(f"   ✓ Saved: table5_5_caries_prevalence_treatment_{timestamp}.csv")
print()

# Consolidate all tidy post-hoc results
all_tidy_posthoc = table3_tidy_posthoc + table5_tidy_posthoc + table5_5_tidy_posthoc
if all_tidy_posthoc:
    consolidated_df = pd.DataFrame(all_tidy_posthoc)
    consolidated_df.to_csv(f'{output_dir}posthoc_pairwise_consolidated_summary_{timestamp}.csv', index=False)
    print(f"   ✓ Saved: posthoc_pairwise_consolidated_summary_{timestamp}.csv")

print("5.3 Creating Table 6: DMFT by Dentition Type and Abuse Type...")
table6 = create_table6_dmft_by_dentition_abuse(df)
if not table6.empty:
    table6.to_csv(f'{output_dir}table6_dmft_dentition_abuse_{timestamp}.csv', index=False)
    print(f"   ✓ Saved: table6_dmft_dentition_abuse_{timestamp}.csv")
print()

print("5.4 Creating Pairwise Mann-Whitney U Tests by Dentition Type...")
table7 = analyze_dmft_by_dentition_with_pairwise(df)
if not table7.empty:
    table7.to_csv(f'{output_dir}table7_pairwise_mannwhitney_dentition_{timestamp}.csv', index=False)
    print(f"   ✓ Saved: table7_pairwise_mannwhitney_dentition_{timestamp}.csv")
print()

print("6. Creating visualizations...")
create_visualizations(df, output_dir)
print()

print("6.1 Creating Pairwise Results...")
plot_boxplot_with_dunn(df, 'DMFT_Index', group_col='abuse', yaxis_name='dmft&DMFT Index', output_dir=output_dir)
plot_boxplot_with_dunn(df, 'Baby_DMFT', group_col='abuse', yaxis_name='Baby DMFT', output_dir=output_dir)
plot_boxplot_with_dunn(df, 'Baby_d', group_col='abuse', yaxis_name='Baby d', output_dir=output_dir)
plot_boxplot_with_dunn(df, 'Healthy_Rate', group_col='abuse', yaxis_name='Healthy Rate', output_dir=output_dir)
plot_boxplot_with_dunn(df, 'Care_Index', group_col='abuse', yaxis_name='Care Index', output_dir=output_dir)
print()

print("6.2 Creating Dentition Type Pairwise Plot...")
plot_boxplot_by_dentition_type(df, output_dir)
print()

print("7. Generating summary report...")
generate_summary_report(df, table3_overall, output_dir, timestamp)
print()

print("=" * 70)
print("Analysis Complete!")
print(f"All outputs saved to: {output_dir}")
print("=" * 70)







