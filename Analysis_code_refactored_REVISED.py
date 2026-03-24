# ============================================================================
# Refactored Analysis Code (REVISED)
# - Enforces a single index observation per child (if an ID column exists)
# - Adds robust handling for ratio indices (Care_Index / UTN_Score)
# - Adds year variable for optional year fixed effects in regression
# - Adds sensitivity analysis including multi-type maltreatment records (abuse_num>1)
# - Adds stratified logistic regression by dentition_type
# ============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Import revised functions
try:
    from Functions_refactored_REVISED import (
        save_value_counts_summary,
        create_table1_demographics,
        create_table2_oral_health_descriptive,
        create_table3_statistical_comparisons,
        create_table4_multivariate_analysis,
        create_forest_plot_vertical,
        create_table5_dmft_by_lifestage_abuse,
        create_table5_5_caries_prevalence_treatment,
        create_table6_dmft_by_dentition_abuse,
        plot_overall_dentition_refined,
        plot_abuse_by_dentition_facet_refined,
        analyze_dmft_by_dentition_with_pairwise,
        create_visualizations,
        plot_boxplot_with_dunn,
        plot_boxplot_by_dentition_type,
        generate_summary_report,
        create_table_dmft_by_year_abuse
    )
except ImportError:
    # If running from a different directory, append path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Functions_refactored_REVISED import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _deduplicate_to_first_exam(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Keep the earliest exam date per child (index exam)."""
    if id_col is None or id_col not in df.columns or 'date' not in df.columns:
        return df

    df_sorted = df.sort_values(['date'])
    before = len(df_sorted)
    df_dedup = df_sorted.drop_duplicates(subset=[id_col], keep='first').copy()
    after = len(df_dedup)
    logger.info(f"Deduplication by {id_col}: {before} -> {after} rows (kept first exam date).")

    return df_dedup


def _engineer_oral_health_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized derivation of DMFT/dmft components and core indices."""
    df = df.copy()

    # Age group
    if 'age_year' in df.columns:
        df['age_group'] = pd.cut(
            df['age_year'],
            bins=[0, 6, 12, 18],
            labels=['Early Childhood (2-6)', 'Middle Childhood (7-12)', 'Adolescence (13-18)'],
            right=True
        )

    # Teeth Columns (if present)
    perm_teeth_cols = [f'U{i}{j}' for i in [1, 2] for j in range(1, 8)] + [f'L{i}{j}' for i in [3, 4] for j in range(1, 8)]
    baby_teeth_cols = [f'u{i}{j}' for i in [5, 6] for j in range(1, 6)] + [f'l{i}{j}' for i in [7, 8] for j in range(1, 6)]

    perm_cols = [c for c in perm_teeth_cols if c in df.columns]
    baby_cols = [c for c in baby_teeth_cols if c in df.columns]

    # Coding:
    # -1: 未萌出、0: 健全、1: 処置歯、2: C0、3: C、4: 喪失歯、5: その他過剰歯等、6: 先天性欠損、7: 歯牙破折、8: 乳歯晩期残存、9: 癒合歯
    if perm_cols:
        # すべての要素が NaN かどうかを判定するためのマスクを作成
        # (すべてが NaN の行は True、一つでもデータがあれば False)
        all_nan_mask_perm = df[perm_cols].isna().all(axis=1)

        df['Perm_D'] = (df[perm_cols] == 3).sum(axis=1).where(~all_nan_mask_perm, np.nan)
        df['Perm_M'] = (df[perm_cols] == 4).sum(axis=1).where(~all_nan_mask_perm, np.nan)
        df['Perm_F'] = (df[perm_cols] == 1).sum(axis=1).where(~all_nan_mask_perm, np.nan)
        df['Perm_Sound'] = (df[perm_cols] == 0).sum(axis=1).where(~all_nan_mask_perm, np.nan)
        df['Perm_DMFT'] = df['Perm_D'] + df['Perm_M'] + df['Perm_F']
        df['Perm_C0'] = (df[perm_cols] == 2).sum(axis=1).where(~all_nan_mask_perm, np.nan)
        df['Perm_DMFT_C0'] = df['Perm_DMFT'] + df['Perm_C0']
        df['Perm_total_teeth'] = ((df[perm_cols].notna()) & (df[perm_cols] != -1)).sum(axis=1)
        df['Perm_sound_rate'] = (df['Perm_Sound'] / df['Perm_total_teeth'] * 100).replace([np.inf, -np.inf], np.nan)
    # else:
    #     for col in ['Perm_D', 'Perm_M', 'Perm_F', 'Perm_Sound', 'Perm_DMFT', 'Perm_C0', 'Perm_DMFT_C0', 'Perm_total_teeth', 'Perm_sound_rate']:
    #         df[col] = np.nan

    
    if baby_cols:
        # すべての要素が NaN かどうかを判定するためのマスクを作成
        # (すべてが NaN の行は True、一つでもデータがあれば False)
        all_nan_mask_baby = df[baby_cols].isna().all(axis=1)

        df['Baby_d'] = (df[baby_cols] == 3).sum(axis=1).where(~all_nan_mask_baby, np.nan)
        df['Baby_m'] = (df[baby_cols] == 4).sum(axis=1).where(~all_nan_mask_baby, np.nan)
        df['Baby_f'] = (df[baby_cols] == 1).sum(axis=1).where(~all_nan_mask_baby, np.nan)
        df['Baby_sound'] = (df[baby_cols] == 0).sum(axis=1).where(~all_nan_mask_baby, np.nan)
        df['Baby_DMFT'] = df['Baby_d'] + df['Baby_m'] + df['Baby_f']
        df['Baby_C0'] = (df[baby_cols] == 2).sum(axis=1).where(~all_nan_mask_baby, np.nan)
        df['Baby_DMFT_C0'] = df['Baby_DMFT'] + df['Baby_C0']
        df['Baby_total_teeth'] = ((df[baby_cols].notna()) & (df[baby_cols] != -1)).sum(axis=1)
        df['Baby_sound_rate'] = (df['Baby_sound'] / df['Baby_total_teeth'] * 100).replace([np.inf, -np.inf], np.nan)
    # else:
    #     for col in ['Baby_d', 'Baby_m', 'Baby_f', 'Baby_sound', 'Baby_DMFT', 'Baby_C0', 'Baby_DMFT_C0', 'Baby_total_teeth', 'Baby_sound_rate']:
    #         df[col] = np.nan

    # Total
    df['DMFT_Index'] = df['Perm_DMFT'].add(df['Baby_DMFT'], fill_value=0)
    df['DMFT_C0'] = df['Perm_DMFT_C0'].add(df['Baby_DMFT_C0'], fill_value=0)
    df['C0_Count'] = df['Perm_C0'].add(df['Baby_C0'], fill_value=0)

    # Indices (explicitly undefined when DMFT_Index == 0)
    denom = df['DMFT_Index'].astype(float)
    filled_total = df['Perm_F'].add(df['Baby_f'], fill_value=0).astype(float)
    df['filled_total'] = filled_total
    decayed_total = df['Perm_D'].add(df['Baby_d'], fill_value=0).astype(float)
    df['decayed_total'] = decayed_total
    missing_total = df['Perm_M'].add(df['Baby_m'], fill_value=0).astype(float)
    df['missing_total'] = missing_total

    df['Care_Index'] = (filled_total / denom * 100).replace([np.inf, -np.inf], np.nan)
    df.loc[denom <= 0, 'Care_Index'] = np.nan  # explicit

    df['UTN_Score'] = (decayed_total / denom * 100).replace([np.inf, -np.inf], np.nan)
    df.loc[denom <= 0, 'UTN_Score'] = np.nan  # explicit

    df['total_teeth'] = df['Perm_total_teeth'] + df['Baby_total_teeth']
    df['Healthy_Rate'] = ((df['Perm_Sound'].add(df['Baby_sound'], fill_value=0)) / df['total_teeth'] * 100).replace([np.inf, -np.inf], np.nan)
    df.loc[df['total_teeth'] <= 0, 'Healthy_Rate'] = np.nan

    # Aliases for downstream functions
    df['Present_Teeth'] = df['total_teeth']
    df['Present_Perm_Teeth'] = df['Perm_total_teeth']
    df['Present_Baby_Teeth'] = df['Baby_total_teeth']

    # Binary outcomes
    df['has_caries'] = (df['DMFT_Index'] > 0).astype(int)
    df['has_untreated_caries'] = (decayed_total > 0).astype(int)

    # Dentition type: 晩期残存は混合歯列になる
    def get_dentition_type(row):
        present_teeth = row['total_teeth'] if pd.notna(row['total_teeth']) else 0
        present_baby = row['Baby_total_teeth'] if pd.notna(row['Baby_total_teeth']) else 0
        present_perm = row['Perm_total_teeth'] if pd.notna(row['Perm_total_teeth']) else 0
        if present_teeth == 0:
            return 'No_Teeth'
        elif present_baby == present_teeth and present_perm == 0:
            return 'primary_dentition'
        elif present_perm == present_teeth and present_baby == 0:
            return 'permanent_dentition'
        else:
            return 'mixed_dentition'

    df['dentition_type'] = df.apply(get_dentition_type, axis=1)

    # Year (for optional year fixed effects)
    if 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
        df['year'] = df['date'].dt.year
    
    df.to_csv('/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data/df.csv', index=False)

    return df


def main():
    logger.info("Starting Analysis...")
    timestamp = datetime.now().strftime('%Y%m%d')

    # ============================================================================
    # Configuration
    # ============================================================================
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # assuming code/ is one level deep
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_DESCRIPTION_OUTPUT_DIR = os.path.join(DATA_DIR, 'data_description')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'result', timestamp) + os.sep

    os.makedirs(DATA_DESCRIPTION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    ORIGINAL_DATA_NAME = 'analysisData_20260211'
    ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, f'{ORIGINAL_DATA_NAME}.csv')

    END_DATE = '2024-03-31'
    target_abuse_types = ["Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse"]

    # Candidate columns for subject ID and examiner (if present)
    SUBJECT_ID_COL_CANDIDATES = ['No_All', 'child_id', 'subject_id', 'case_id', 'ID', 'id']
    EXAMINER_COL_CANDIDATES = ['dentist', 'examiner', 'doctor', 'operator', 'checker']

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
    if 'date' in data0.columns:
        data0['date'] = pd.to_datetime(data0['date'], errors='coerce')

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

    # Set categorical order
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

    save_value_counts_summary(
        data0,
        os.path.join(DATA_DESCRIPTION_OUTPUT_DIR, f'unique_values_summary_{ORIGINAL_DATA_NAME}.csv'),
        exclude_cols=["No_All", "instruction_detail", "instruction", "memo"]
    )
    data0.describe().to_csv(os.path.join(DATA_DESCRIPTION_OUTPUT_DIR, f'{ORIGINAL_DATA_NAME}_description.csv'))

    # ============================================================================
    # Filtering & Flow accounting
    # ============================================================================
    logger.info("Filtering data...")
    df_date = data0.copy()
    if 'date' in df_date.columns:
        df_date = df_date[df_date['date'] <= END_DATE].copy()

    # all target abuse with at least one abuse type recorded (for sensitivity / flow)
    if 'abuse_num' in df_date.columns:
        df_all = df_date[(df_date['abuse_num'] >= 1) & (df_date['abuse'].isin(target_abuse_types))].copy()
    else:
        df_all = df_date[df_date['abuse'].isin(target_abuse_types)].copy()

    # Main analysis = single primary abuse type only
    if 'abuse_num' in df_all.columns:
        df_main = df_all[df_all['abuse_num'] == 1].copy()
    else:
        df_main = df_all.copy()

    # Identify subject ID column (if any) and deduplicate to first exam
    subject_id_col = _pick_first_existing_col(df_main, SUBJECT_ID_COL_CANDIDATES)
    examiner_col = _pick_first_existing_col(df_main, EXAMINER_COL_CANDIDATES)

    df_main = _deduplicate_to_first_exam(df_main, subject_id_col)

    # Keep consistent categories
    if 'abuse' in df_main.columns and hasattr(df_main['abuse'], 'cat'):
        df_main['abuse'] = df_main['abuse'].cat.remove_unused_categories()

    logger.info(f"Main dataset shape (single-type + dedup if possible): {df_main.shape}")

    # Save filtered main dataset
    csv_name = f"{ORIGINAL_DATA_NAME}_tillMar2024_singleType_dedup"
    df_main.to_csv(os.path.join(DATA_DIR, f'{csv_name}.csv'), index=False)

    # Flow summary (helps Methods transparency)
    flow_rows = []
    flow_rows.append({'Step': 'Loaded raw', 'N': int(len(data0))})
    flow_rows.append({'Step': f'Date <= {END_DATE}', 'N': int(len(df_date))})
    flow_rows.append({'Step': 'Target maltreatment (abuse in 4 types) & abuse_num>=1', 'N': int(len(df_all))})
    if 'abuse_num' in df_all.columns:
        flow_rows.append({'Step': 'Single-type only (abuse_num==1)', 'N': int((df_all['abuse_num'] == 1).sum())})
        flow_rows.append({'Step': 'Multi-type excluded (abuse_num>1)', 'N': int((df_all['abuse_num'] > 1).sum())})
    if subject_id_col:
        flow_rows.append({'Step': f'Deduplicated to first exam per {subject_id_col}', 'N': int(len(df_main))})
    pd.DataFrame(flow_rows).to_csv(os.path.join(OUTPUT_DIR, f'flow_summary_{timestamp}.csv'), index=False)

    # Save a compact profile of excluded multi-type cases (for supplemental)
    if 'abuse_num' in df_all.columns:
        df_multi = df_all[df_all['abuse_num'] > 1].copy()
        if not df_multi.empty:
            df_multi_prof = _engineer_oral_health_variables(df_multi)
            prof_cols = [c for c in ['age_year', 'sex', 'abuse', 'abuse_num', 'DMFT_Index', 'Care_Index', 'UTN_Score', 'Healthy_Rate'] if c in df_multi_prof.columns]
            df_multi_prof[prof_cols].describe(include='all').to_csv(os.path.join(OUTPUT_DIR, f'multitype_profile_{timestamp}.csv'))

    # ============================================================================
    # Feature engineering (main)
    # ============================================================================
    logger.info("Calculating derived variables (main)...")
    df = _engineer_oral_health_variables(df_main)

    # ============================================================================
    # Analysis & Reporting (main)
    # ============================================================================
    logger.info("Running statistical analysis (main)...")

    # Table 1
    table1 = create_table1_demographics(df)
    table1.to_csv(os.path.join(OUTPUT_DIR, f'table1_demographics_{timestamp}.csv'), index=False)

    # Table 1 by dentition type
    for dent_type in ['primary_dentition', 'mixed_dentition', 'permanent_dentition']:
        df_dent = df[df['dentition_type'] == dent_type]
        if not df_dent.empty:
            table1_dent = create_table1_demographics(df_dent)
            table1_dent.to_csv(os.path.join(OUTPUT_DIR, f'table1_demographics_{dent_type}_{timestamp}.csv'), index=False)

    # Table 2
    table2_cont, table2_cat = create_table2_oral_health_descriptive(df)
    table2_cont.to_csv(os.path.join(OUTPUT_DIR, f'table2_continuous_{timestamp}.csv'), index=False)
    table2_cat.to_csv(os.path.join(OUTPUT_DIR, f'table2_categorical_{timestamp}.csv'), index=False)

    # Table 3
    t3_overall, t3_posthoc, t3_pairwise, t3_tidy = create_table3_statistical_comparisons(df)
    t3_overall.to_csv(os.path.join(OUTPUT_DIR, f'table3_overall_tests_{timestamp}.csv'), index=False)
    t3_posthoc.to_csv(os.path.join(OUTPUT_DIR, f'table3_posthoc_{timestamp}.csv'), index=False)
    t3_pairwise.to_csv(os.path.join(OUTPUT_DIR, f'table3_pairwise_mw_{timestamp}.csv'), index=False)

    # Table 4 (overall logistic regression; spline age + year FE + examiner FE if available)
    table4_overall = create_table4_multivariate_analysis(
        df,
        use_age_spline=True,
        age_spline_df=4,
        add_year_fe=True,
        year_col='year',
        examiner_col=examiner_col,
        id_col=subject_id_col,
        stratify_by=None
    )
    table4_overall.to_csv(os.path.join(OUTPUT_DIR, f'table4_logistic_regression_{timestamp}.csv'), index=False)

    # Table 4b (stratified by dentition_type)
    table4_dent = create_table4_multivariate_analysis(
        df,
        use_age_spline=True,
        age_spline_df=4,
        add_year_fe=True,
        year_col='year',
        examiner_col=examiner_col,
        id_col=subject_id_col,
        stratify_by='dentition_type',
        strata_order=['mixed_dentition', 'primary_dentition', 'permanent_dentition']
    )
    if not table4_dent.empty:
        table4_dent.to_csv(os.path.join(OUTPUT_DIR, f'table4_logistic_regression_by_dentition_{timestamp}.csv'), index=False)

    # Forest plot for overall table 4
    create_forest_plot_vertical(table4_overall, df, OUTPUT_DIR, timestamp)

    # Table 5
    table5, t5_tidy = create_table5_dmft_by_lifestage_abuse(df)
    if not table5.empty:
        table5.to_csv(os.path.join(OUTPUT_DIR, f'table5_dmft_lifestage_abuse_{timestamp}.csv'), index=False)

    # Table 5.5
    table5_5, t5_5_tidy = create_table5_5_caries_prevalence_treatment(df)
    if not table5_5.empty:
        table5_5.to_csv(os.path.join(OUTPUT_DIR, f'table5_5_caries_prevalence_treatment_{timestamp}.csv'), index=False)

    # Table 6
    t6_summary, t6_within_dentition, t6_within_abuse, t6_overall_dentition = create_table6_dmft_by_dentition_abuse(df)
    if not t6_summary.empty:
        t6_summary.to_csv(os.path.join(OUTPUT_DIR, f'table6_dmft_dentition_abuse_{timestamp}.csv'), index=False)
    if not t6_within_dentition.empty:
        t6_within_dentition.to_csv(os.path.join(OUTPUT_DIR, f'table6_within_dentition_posthoc_{timestamp}.csv'), index=False)
    if not t6_within_abuse.empty:
        t6_within_abuse.to_csv(os.path.join(OUTPUT_DIR, f'table6_within_abuse_posthoc_{timestamp}.csv'), index=False)
    if not t6_overall_dentition.empty:
        t6_overall_dentition.to_csv(os.path.join(OUTPUT_DIR, f'table6_overall_dentition_posthoc_{timestamp}.csv'), index=False)
    
    # --- 生成图 1：整体牙列对比 ---
    # 传入 t6_overall_dentition 进行显著性标注
    plot_overall_dentition_refined(
        df=df, 
        posthoc_df=t6_overall_dentition, 
        y_col='DMFT_Index',
        ylabel='Caries Experience',
        save_path=os.path.join(OUTPUT_DIR, f'figure_overall_dentition_{timestamp}.png')
    )

    # --- 生成图 2：分阶段受虐对比 ---
    # 传入 t6_within_dentition 进行显著性标注
    plot_abuse_by_dentition_facet_refined(
        df=df, 
        posthoc_df=t6_within_dentition, 
        y_col='DMFT_Index',
        ylabel='Caries Experience',
        save_path=os.path.join(OUTPUT_DIR, f'figure_abuse_by_dentition_facet_{timestamp}.png')
    )

    # Table 7 (year x abuse)
    table7 = create_table_dmft_by_year_abuse(df)
    if not table7.empty:
        table7.to_csv(os.path.join(OUTPUT_DIR, f'table7_dmft_by_year_abuse_{timestamp}.csv'), index=False)

    # Visualization
    create_visualizations(df, OUTPUT_DIR)

    # Pairwise plots (note: Care_Index plotted among DMFT>0 only in revised function)
    for var in ['DMFT_Index', 'Healthy_Rate', 'Baby_d', 'Baby_DMFT', 'Care_Index', 'UTN_Score']:
        try:
            plot_boxplot_with_dunn(df, var, group_col='abuse', ylabel=var, output_dir=OUTPUT_DIR)
        except Exception as e:
            print(f"Error drawing pairwise plot for {var}: {e}")
    plot_boxplot_by_dentition_type(df, output_dir=OUTPUT_DIR)

    # Summary report
    generate_summary_report(df, t3_overall, OUTPUT_DIR, timestamp)

    # ============================================================================
    # Sensitivity analysis: include multi-type (abuse_num>=1), adjust for is_multitype
    # ============================================================================
    if 'abuse_num' in df_all.columns:
        logger.info("Running sensitivity analysis including multi-type cases...")
        df_sens = df_all.copy()

        # Add indicator and deduplicate (same rule) if possible
        df_sens['is_multitype'] = (df_sens['abuse_num'] > 1).astype(int)
        df_sens = _deduplicate_to_first_exam(df_sens, subject_id_col)

        if 'abuse' in df_sens.columns and hasattr(df_sens['abuse'], 'cat'):
            df_sens['abuse'] = df_sens['abuse'].cat.remove_unused_categories()

        df_sens = _engineer_oral_health_variables(df_sens)

        # Logistic regression with additional covariate is_multitype
        table4_sens = create_table4_multivariate_analysis(
            df_sens,
            use_age_spline=True,
            age_spline_df=4,
            add_year_fe=True,
            year_col='year',
            examiner_col=examiner_col,
            id_col=subject_id_col,
            add_covariates=['is_multitype']
        )
        if not table4_sens.empty:
            table4_sens.to_csv(os.path.join(OUTPUT_DIR, f'table4_logistic_regression_sensitivity_multitype_{timestamp}.csv'), index=False)

    logger.info(f"Analysis complete. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()