
# ============================================================================
# Refactored Analysis Code
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Import refactored functions
try:
    from Functions_refactored import (
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
        generate_summary_report
    )
except ImportError:
    # If running from a different directory, append path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Functions_refactored import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Analysis...")
    timestamp = datetime.now().strftime('%Y%m%d')

    # ============================================================================
    # Configuration
    # ============================================================================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # assuming code/ is one level deep
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_DESCRIPTION_OUTPUT_DIR = os.path.join(DATA_DIR, 'data_description')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'result', timestamp)

    # Ensure directories exist
    os.makedirs(DATA_DESCRIPTION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ORIGINAL_DATA_NAME = 'analysisData_20260211'
    ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, f'{ORIGINAL_DATA_NAME}.csv')

    # ============================================================================
    # Data Loading
    # ============================================================================
    logger.info(f"Loading data from {ORIGINAL_DATA_PATH}")
    if not os.path.exists(ORIGINAL_DATA_PATH):
        logger.error(f"Data file not found: {ORIGINAL_DATA_PATH}")
        return

    data0 = pd.read_csv(ORIGINAL_DATA_PATH)
    logger.info(f"Loaded data shape: {data0.shape}")

    # Save columns
    with open(os.path.join(DATA_DESCRIPTION_OUTPUT_DIR, f'{ORIGINAL_DATA_NAME}_colnames.txt'), 'w') as f:
        for col in data0.columns:
            f.write(f"{col}\n")

    # ============================================================================
    # Preprocessing
    # ============================================================================
    data0['date'] = pd.to_datetime(data0['date'])

    # Mappings
    mappings = {
        'abuse': {1: "Physical Abuse", 2: "Neglect", 3: "Emotional Abuse", 4: "Sexual Abuse", 5: "Delinquency", 6: "Parenting Difficulties", 7: "Others"},
        'occlusalRelationship': {1: "Normal Occlusion", 2: "Crowding", 3: "Anterior Crossbite", 4: "Open Bite", 5: "Maxillary Protrusion", 6: "Crossbite", 7: "Others"},
        'needTOBEtreated': {1: "No Treatment Required", 2: "Treatment Required"},
        'emergency': {1: "Urgent Treatment Required"},
        'gingivitis': {1: "No Gingivitis", 2: "Gingivitis"},
        'OralCleanStatus': {1: "Poor", 2: "Fair", 3: "Good"},
        'habits': {1: "None", 2: "Digit Sucking", 3: "Nail biting", 4: "Tongue Thrusting", 5: "Smoking", 6: "Others"}
    }

    for col, mapping in mappings.items():
        if col in data0.columns:
            data0[col] = data0[col].replace(mapping)

    # Set Categorical Order
    orders = {
        'abuse': ["Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse", "Delinquency", "Parenting Difficulties", "Others"],
        'occlusalRelationship': ["Normal Occlusion", "Crowding", "Anterior Crossbite", "Open Bite", "Maxillary Protrusion", "Crossbite", "Others"],
        'needTOBEtreated': ["No Treatment Required", "Treatment Required"],
        'emergency': ["Urgent Treatment Required"],
        'gingivitis': ["No Gingivitis", "Gingivitis"],
        'OralCleanStatus': ["Poor", "Fair", "Good"],
        'habits': ["None", "Digit Sucking", "Nail biting", "Tongue Thrusting", "Smoking", "Others"]
    }

    for col, order in orders.items():
        if col in data0.columns:
            data0[col] = pd.Categorical(data0[col], categories=order, ordered=True)

    # Save cleaned data
    cleaned_path = os.path.join(DATA_DIR, f'{ORIGINAL_DATA_NAME}_AllData_cleaned.csv')
    data0.to_csv(cleaned_path, index=False)
    
    # Save Value Counts
    save_value_counts_summary(data0, os.path.join(DATA_DESCRIPTION_OUTPUT_DIR, f'unique_values_summary_{ORIGINAL_DATA_NAME}.csv'), 
                              exclude_cols=["No_All", "instruction_detail", "instruction", "memo"])
    data0.describe().to_csv(os.path.join(DATA_DESCRIPTION_OUTPUT_DIR, f'{ORIGINAL_DATA_NAME}_description.csv'))

    # ============================================================================
    # Filtering
    # ============================================================================
    logger.info("Filtering data...")
    # Filter by date
    df = data0[data0['date'] <= '2024-03-31'].copy()
    
    # Filter by abuse criteria
    # abuse_num != 0 AND abuse_num == 1 AND abuse is target
    target_abuse_types = ["Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse"]
    df = df[(df['abuse_num'] == 1) & (df['abuse'].isin(target_abuse_types))].copy()
    
    # Clean unused categories
    df['abuse'] = df['abuse'].cat.remove_unused_categories()
    
    logger.info(f"Filtered data shape: {df.shape}")
    
    # Save filtered
    csv_name = f"{ORIGINAL_DATA_NAME}_tillMar2024_noAbuseNA_Only1"
    df.to_csv(os.path.join(DATA_DIR, f'{csv_name}.csv'), index=False)
    save_value_counts_summary(df, os.path.join(DATA_DESCRIPTION_OUTPUT_DIR, f'unique_values_summary_{csv_name}.csv'), 
                              exclude_cols=["No_All", "instruction_detail", "instruction", "memo"])

    # ============================================================================
    # Feature Engineering (Vectorized)
    # ============================================================================
    logger.info("Calculating derived variables...")
    
    # Age Group
    df['age_group'] = pd.cut(df['age_year'], bins=[0, 6, 12, 18], 
                             labels=['Early Childhood (2-6)', 'Middle Childhood (7-12)', 'Adolescence (13-18)'], 
                             right=True)

    # Teeth Columns
    perm_teeth_cols = [f'U{i}{j}' for i in [1, 2] for j in range(1, 8)] + [f'L{i}{j}' for i in [3, 4] for j in range(1, 8)]
    baby_teeth_cols = [f'u{i}{j}' for i in [5, 6] for j in range(1, 6)] + [f'l{i}{j}' for i in [7, 8] for j in range(1, 6)]
    
    perm_cols = [c for c in perm_teeth_cols if c in df.columns]
    baby_cols = [c for c in baby_teeth_cols if c in df.columns]

    # Vectorized DMFT Calculation
    # Permanent
    if perm_cols:
        df['Perm_D'] = (df[perm_cols] == 3).sum(axis=1)
        df['Perm_M'] = (df[perm_cols] == 4).sum(axis=1)
        df['Perm_F'] = (df[perm_cols] == 1).sum(axis=1)
        df['Perm_Sound'] = (df[perm_cols] == 0).sum(axis=1)
        df['Perm_DMFT'] = df['Perm_D'] + df['Perm_M'] + df['Perm_F']
        df['Perm_C0'] = (df[perm_cols] == 2).sum(axis=1)
        df['Perm_DMFT_C0'] = df['Perm_DMFT'] + df['Perm_C0']
        df["Perm_total_teeth"] = ((df[perm_cols].notna()) & (df[perm_cols] != -1)).sum(axis=1)
        df["Perm_sound_rate"] = (df["Perm_Sound"] / df["Perm_total_teeth"] * 100).replace([np.inf, -np.inf], np.nan)
    else:
        for col in ['Perm_D', 'Perm_M', 'Perm_F', 'Perm_Sound', 'Perm_DMFT', 'Perm_C0', 'Perm_DMFT_C0', 'Perm_total_teeth', 'Perm_sound_rate']:
            df[col] = 0

    # Baby
    if baby_cols:
        df['Baby_d'] = (df[baby_cols] == 3).sum(axis=1)
        df['Baby_m'] = (df[baby_cols] == 4).sum(axis=1)
        df['Baby_f'] = (df[baby_cols] == 1).sum(axis=1)
        df['Baby_sound'] = (df[baby_cols] == 0).sum(axis=1)
        df['Baby_DMFT'] = df['Baby_d'] + df['Baby_m'] + df['Baby_f']
        df['Baby_C0'] = (df[baby_cols] == 2).sum(axis=1)
        df['Baby_DMFT_C0'] = df['Baby_DMFT'] + df['Baby_C0']
        df["Baby_total_teeth"] = ((df[baby_cols].notna()) & (df[baby_cols] != -1)).sum(axis=1)
        df["Baby_sound_rate"] = (df["Baby_sound"] / df["Baby_total_teeth"] * 100).replace([np.inf, -np.inf], np.nan)
    else:
        for col in ['Baby_d', 'Baby_m', 'Baby_f', 'Baby_sound', 'Baby_DMFT', 'Baby_C0', 'Baby_DMFT_C0', 'Baby_total_teeth', 'Baby_sound_rate']:
            df[col] = 0

    # Total
    df['DMFT_Index'] = df['Perm_DMFT'] + df['Baby_DMFT']
    df['DMFT_Index_C0'] = df['Perm_DMFT_C0'] + df['Baby_DMFT_C0']
    
    # Indices
    df['Care_Index'] = ((df['Perm_F'] + df['Baby_f']) / df['DMFT_Index'] * 100).replace([np.inf, -np.inf], np.nan)
    df['total_teeth'] = df['Perm_total_teeth'] + df['Baby_total_teeth']
    df['Healthy_Rate'] = ((df['Perm_Sound'] + df['Baby_sound']) / df['total_teeth'] * 100).replace([np.inf, -np.inf], np.nan)
    
    # Aliases for functions expecting different names
    df['Present_Teeth'] = df['total_teeth']
    df['Present_Perm_Teeth'] = df['Perm_total_teeth']
    df['Present_Baby_Teeth'] = df['Baby_total_teeth']
    
    # Indicators
    df['has_caries'] = (df['DMFT_Index'] > 0).astype(int)
    df['has_untreated_caries'] = ((df['Perm_D'] + df['Baby_d']) > 0).astype(int)

    # Basic stats
    logger.info(f"DMFT Mean: {df['DMFT_Index'].mean():.2f}")

    # ============================================================================
    # Analysis & Reporting
    # ============================================================================
    logger.info("Runnning statistical analysis...")
    
    # Table 1
    table1 = create_table1_demographics(df)
    table1.to_csv(os.path.join(OUTPUT_DIR, f'table1_demographics_{timestamp}.csv'), index=False)
    
    # Table 2
    table2_cont, table2_cat = create_table2_oral_health_descriptive(df)
    table2_cont.to_csv(os.path.join(OUTPUT_DIR, f'table2_continuous_{timestamp}.csv'), index=False)
    table2_cat.to_csv(os.path.join(OUTPUT_DIR, f'table2_categorical_{timestamp}.csv'), index=False)
    
    # Table 3
    t3_overall, t3_posthoc, t3_pairwise, t3_tidy = create_table3_statistical_comparisons(df)
    t3_overall.to_csv(os.path.join(OUTPUT_DIR, f'table3_overall_tests_{timestamp}.csv'), index=False)
    t3_posthoc.to_csv(os.path.join(OUTPUT_DIR, f'table3_posthoc_{timestamp}.csv'), index=False)
    
    # Table 4
    table4 = create_table4_multivariate_analysis(df)
    table4.to_csv(os.path.join(OUTPUT_DIR, f'table4_logistic_regression_{timestamp}.csv'), index=False)
    
    # Forest Plot
    create_forest_plot_vertical(table4, df, OUTPUT_DIR, timestamp)
    
    # Table 5
    table5, t5_tidy = create_table5_dmft_by_lifestage_abuse(df)
    if not table5.empty:
        table5.to_csv(os.path.join(OUTPUT_DIR, f'table5_dmft_lifestage_abuse_{timestamp}.csv'), index=False)
        
    # Table 5.5
    table5_5, t5_5_tidy = create_table5_5_caries_prevalence_treatment(df)
    if not table5_5.empty:
        table5_5.to_csv(os.path.join(OUTPUT_DIR, f'table5_5_caries_prevalence_treatment_{timestamp}.csv'), index=False)
        
    # Table 6
    table6 = create_table6_dmft_by_dentition_abuse(df)
    if not table6.empty:
        table6.to_csv(os.path.join(OUTPUT_DIR, f'table6_dmft_dentition_abuse_{timestamp}.csv'), index=False)

    # Visualization
    create_visualizations(df, OUTPUT_DIR)
    
    # Pairwise plots
    for var in ['DMFT_Index', 'Baby_DMFT', 'Baby_d', 'Healthy_Rate', 'Care_Index']:
        plot_boxplot_with_dunn(df, var, group_col='abuse', yaxis_name=var, output_dir=OUTPUT_DIR)
        
    plot_boxplot_by_dentition_type(df, output_dir=OUTPUT_DIR)
    
    # Report
    generate_summary_report(df, t3_overall, OUTPUT_DIR, timestamp)
    
    logger.info(f"Analysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
