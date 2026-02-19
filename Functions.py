import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
import scikit_posthocs as sp
from scikit_posthocs import posthoc_dunn
import statsmodels.api as sm
import itertools
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d')
OUTPUT_DIR = './'

"""
Value Counts Summary
"""
def save_value_counts_summary(df, output_path, exclude_cols=None):
    """
    Calculates value counts for each column in the dataframe strings and saves the summary to a CSV file.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        output_path (str): Path to save the CSV output.
        exclude_cols (list, optional): List of column names to exclude. Defaults to None.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    results = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        value_counts = df[col].value_counts(dropna=False)
        
        for val, count in value_counts.items():
            results.append({
                "Column": col,
                "Value": val,
                "Count": count
            })
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Summary saved to {output_path}")

"""
Table 1: 虐待分類別の人口統計学的特性
"""
def create_table1_demographics(df):
    abuse_types = df['abuse'].cat.categories
    results = []
    
    # 総数
    total_row = {'Variable': 'Total N', 'Category': ''}
    for abuse in abuse_types:
        n = df[df['abuse'] == abuse].shape[0]
        total_row[abuse] = str(n)
    total_row['Total'] = str(df.shape[0])
    total_row['p-value'] = ''
    results.append(total_row)
    
    # 性別分布
    sex_row_header = {'Variable': 'Sex', 'Category': '', **{abuse: '' for abuse in abuse_types}, 
                      'Total': '', 'p-value': ''}
    results.append(sex_row_header)
    
    contingency_sex = pd.crosstab(df['abuse'], df['sex'])
    chi2_sex, p_sex, _, _ = chi2_contingency(contingency_sex)
    
    for sex in ['Male', 'Female']:
        row = {'Variable': '', 'Category': f'  {sex}'}
        for abuse in abuse_types:
            n = df[(df['abuse'] == abuse) & (df['sex'] == sex)].shape[0]
            total_abuse = df[df['abuse'] == abuse].shape[0]
            pct = (n / total_abuse * 100) if total_abuse > 0 else 0
            row[abuse] = f"{n} ({pct:.1f}%)"
        
        total_n = df[df['sex'] == sex].shape[0]
        total_pct = total_n / df.shape[0] * 100
        row['Total'] = f"{total_n} ({total_pct:.1f}%)"
        row['p-value'] = f"{p_sex:.3f}" if sex == 'Male' else ''
        results.append(row)
    
    # 年齢（連続変数）
    age_row = {'Variable': 'Age (years)', 'Category': 'Mean ± SD'}
    for abuse in abuse_types:
        subset = df[df['abuse'] == abuse]['age_year']
        age_row[abuse] = f"{subset.mean():.1f} ± {subset.std():.1f}"
    total_age = df['age_year']
    age_row['Total'] = f"{total_age.mean():.1f} ± {total_age.std():.1f}"
    
    groups = [df[df['abuse'] == abuse]['age_year'].dropna() for abuse in abuse_types]
    _, p_age = kruskal(*groups)
    age_row['p-value'] = f"{p_age:.3f}"
    results.append(age_row)
    
    # 年齢（中央値・四分位範囲）
    age_median_row = {'Variable': '', 'Category': 'Median [IQR]'}
    for abuse in abuse_types:
        subset = df[df['abuse'] == abuse]['age_year']
        q25, q50, q75 = subset.quantile([0.25, 0.5, 0.75])
        age_median_row[abuse] = f"{q50:.0f} [{q25:.0f}-{q75:.0f}]"
    q25, q50, q75 = df['age_year'].quantile([0.25, 0.5, 0.75])
    age_median_row['Total'] = f"{q50:.0f} [{q25:.0f}-{q75:.0f}]"
    age_median_row['p-value'] = ''
    results.append(age_median_row)
    
    # 年齢グループ
    if 'age_group' in df.columns:
        age_group_header = {'Variable': 'Age Group', 'Category': '', **{abuse: '' for abuse in abuse_types},
                            'Total': '', 'p-value': ''}
        results.append(age_group_header)
        
        df_valid = df.dropna(subset=['age_group'])
        contingency_age = pd.crosstab(df_valid['abuse'], df_valid['age_group'])
        chi2_age_grp, p_age_grp, _, _ = chi2_contingency(contingency_age)
        
        first_group = True
        for age_grp in df['age_group'].cat.categories:
            row = {'Variable': '', 'Category': f'  {age_grp}'}
            for abuse in abuse_types:
                n = df[(df['abuse'] == abuse) & (df['age_group'] == age_grp)].shape[0]
                total_abuse = df[df['abuse'] == abuse].shape[0]
                pct = (n / total_abuse * 100) if total_abuse > 0 else 0
                row[abuse] = f"{n} ({pct:.1f}%)"
            
            total_n = df[df['age_group'] == age_grp].shape[0]
            total_pct = total_n / df.shape[0] * 100
            row['Total'] = f"{total_n} ({total_pct:.1f}%)"
            row['p-value'] = f"{p_age_grp:.3f}" if first_group else ''
            first_group = False
            results.append(row)
    
    table1 = pd.DataFrame(results)
    return table1

"""
Table 2: 虐待分類別の口腔内状況（記述統計）
"""
def create_table2_oral_health_descriptive(df):
    abuse_types = df['abuse'].cat.categories
    
    continuous_vars = [
        ('DMFT_Index', 'DMFT Index (Total)'),
        ('Perm_DMFT', 'Permanent DMFT'),
        ('Baby_DMFT', 'Primary dmft'),
        ('Perm_D', 'Permanent D (Decayed)'),
        ('Perm_M', 'Permanent M (Missing)'),
        ('Perm_F', 'Permanent F (Filled)'),
        ('Baby_d', 'Primary d (decayed)'),
        ('Baby_m', 'Primary m (missing)'),
        ('Baby_f', 'Primary f (filled)'),
        ('C0_Count', 'C0 (Incipient Caries)'),
        ('Healthy_Rate', 'Healthy Teeth Rate (%)'),
        ('Care_Index', 'Care Index (%)'),
        ('UTN_Score', 'Untreated Caries Rate (%)'),
        ('Trauma_Count', 'Dental Trauma Count'),
        ('RDT_Count', 'Retained Deciduous Teeth')
    ]
    
    results_continuous = []
    
    for var_name, var_label in continuous_vars:
        if var_name not in df.columns:
            continue
            
        row = {'Variable': var_label}
        
        for abuse in abuse_types:
            subset = df[df['abuse'] == abuse][var_name].dropna()
            if len(subset) > 0:
                mean = subset.mean()
                std = subset.std()
                median = subset.median()
                q25, q75 = subset.quantile([0.25, 0.75])
                row[f'{abuse}_Mean_SD'] = f"{mean:.2f} ± {std:.2f}"
                row[f'{abuse}_Median_IQR'] = f"{median:.1f} [{q25:.1f}-{q75:.1f}]"
            else:
                row[f'{abuse}_Mean_SD'] = 'N/A'
                row[f'{abuse}_Median_IQR'] = 'N/A'
        
        total = df[var_name].dropna()
        if len(total) > 0:
            row['Total_Mean_SD'] = f"{total.mean():.2f} ± {total.std():.2f}"
            row['Total_Median_IQR'] = f"{total.median():.1f} [{total.quantile(0.25):.1f}-{total.quantile(0.75):.1f}]"
        
        groups = [df[df['abuse'] == abuse][var_name].dropna() for abuse in abuse_types]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                _, p_val = kruskal(*groups)
                row['p-value'] = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
            except:
                row['p-value'] = 'N/A'
        else:
            row['p-value'] = 'N/A'
        
        results_continuous.append(row)
    
    categorical_vars = [
        ('gingivitis', 'Gingivitis'),
        ('needTOBEtreated', 'Treatment Need'),
        ('occlusalRelationship', 'Occlusal Relationship'),
        ('OralCleanStatus', 'Oral Hygiene Status'),
        ('habits', 'Oral Habits')
    ]
    
    results_categorical = []
    
    for var_name, var_label in categorical_vars:
        if var_name not in df.columns:
            continue
        
        header_row = {'Variable': var_label, 'Category': ''}
        for abuse in abuse_types:
            header_row[f'{abuse}_n'] = ''
            header_row[f'{abuse}_%'] = ''
        header_row['Total_n'] = ''
        header_row['Total_%'] = ''
        header_row['p-value'] = ''
        results_categorical.append(header_row)
        
        df_valid = df.dropna(subset=[var_name])
        contingency = pd.crosstab(df_valid['abuse'], df_valid[var_name])
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency)
        except:
            p_val = np.nan
        
        categories = df[var_name].cat.categories if hasattr(df[var_name], 'cat') else df[var_name].dropna().unique()
        first_cat = True
        
        for cat in categories:
            row = {'Variable': '', 'Category': f'  {cat}'}
            
            for abuse in abuse_types:
                n = df[(df['abuse'] == abuse) & (df[var_name] == cat)].shape[0]
                total_abuse = df[(df['abuse'] == abuse) & (df[var_name].notna())].shape[0]
                pct = (n / total_abuse * 100) if total_abuse > 0 else 0
                row[f'{abuse}_n'] = n
                row[f'{abuse}_%'] = f"{pct:.1f}"
            
            total_n = df[df[var_name] == cat].shape[0]
            total_valid = df[df[var_name].notna()].shape[0]
            total_pct = (total_n / total_valid * 100) if total_valid > 0 else 0
            row['Total_n'] = total_n
            row['Total_%'] = f"{total_pct:.1f}"
            
            if first_cat and not np.isnan(p_val):
                row['p-value'] = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
            else:
                row['p-value'] = ''
            first_cat = False
            
            results_categorical.append(row)
    
    table2_continuous = pd.DataFrame(results_continuous)
    table2_categorical = pd.DataFrame(results_categorical)
    
    return table2_continuous, table2_categorical

"""
Table 3: 虐待分類間の統計的比較
"""
def create_table3_statistical_comparisons(df):
    """
    Table 3: 虐待分類間の統計的比較
    """
    abuse_types = list(df['abuse'].cat.categories)
    
    continuous_vars = [
        'DMFT_Index', 'Perm_DMFT', 'Baby_DMFT', 
        'Perm_D', 'Perm_M', 'Perm_F',
        'Baby_d', 'Baby_m', 'Baby_f',
        'C0_Count', 'Healthy_Rate', 'Care_Index', 
        'UTN_Score', 'Trauma_Count',"DMFT_C0","Perm_DMFT_C0","Baby_DMFT_C0"
    ]
    
    overall_results = []
    
    for var in continuous_vars:
        if var not in df.columns:
            continue
        
        groups = [df[df['abuse'] == abuse][var].dropna() for abuse in abuse_types]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2:
            continue
        
        try:
            h_stat, p_kw = kruskal(*groups)
            overall_results.append({
                'Variable': var,
                'Test': 'Kruskal-Wallis',
                'Statistic': f"{h_stat:.3f}",
                'p-value': f"{p_kw:.4f}" if p_kw >= 0.0001 else "<0.0001",
                'Significant': 'Yes' if p_kw < 0.05 else 'No'
            })
        except Exception as e:
            overall_results.append({
                'Variable': var,
                'Test': 'Kruskal-Wallis',
                'Statistic': 'N/A',
                'p-value': 'N/A',
                'Significant': 'N/A'
            })
    
    posthoc_results = []
    tidy_posthoc_pairwise = []
    
    for var in continuous_vars:
        if var not in df.columns:
            continue
        
        # Only run Dunn's test if Kruskal-Wallis was significant
        kw_p = next((r['p-value'] for r in overall_results if r['Variable'] == var and r['Significant'] == 'Yes'), None)
        if kw_p is None:
            continue

        try:
            dunn_adj, dunn_unadj = posthoc_dunn(df, val_col=var, group_col='abuse', p_adjust='bonferroni')
            
            for i, abuse1 in enumerate(abuse_types):
                for abuse2 in abuse_types[i+1:]:
                    if abuse1 in dunn_adj.index and abuse2 in dunn_adj.columns:
                        p_adj = dunn_adj.loc[abuse1, abuse2]
                        p_unadj = dunn_unadj.loc[abuse1, abuse2]
                        
                        posthoc_results.append({
                            'Variable': var,
                            'Comparison': f"{abuse1} vs {abuse2}",
                            'p-value (adjusted)': f"{p_adj:.4f}" if p_adj >= 0.0001 else "<0.0001",
                            'Significant': 'Yes' if p_adj < 0.05 else 'No'
                        })
                        
                        # Added tidy summary table for pairwise significance
                        tidy_posthoc_pairwise.append({
                            'variable': var,
                            'group1': abuse1,
                            'group2': abuse2,
                            'p_unadjusted': p_unadj,
                            'p_adjusted': p_adj,
                            'significant': p_adj < 0.05
                        })
        except Exception as e:
            pass
    
    pairwise_results = []
    abuse_pairs = list(itertools.combinations(abuse_types, 2))
    n_comparisons = len(abuse_pairs) * len(continuous_vars)
    bonferroni_threshold = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
    
    for var in continuous_vars:
        if var not in df.columns:
            continue
        
        for abuse1, abuse2 in abuse_pairs:
            group1 = df[df['abuse'] == abuse1][var].dropna()
            group2 = df[df['abuse'] == abuse2][var].dropna()
            
            if len(group1) == 0 or len(group2) == 0:
                continue
            
            try:
                u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                
                n1, n2 = len(group1), len(group2)
                r = 1 - (2 * u_stat) / (n1 * n2)
                
                pairwise_results.append({
                    'Variable': var,
                    'Group1': abuse1,
                    'Group2': abuse2,
                    'Group1_Median': f"{group1.median():.2f}",
                    'Group2_Median': f"{group2.median():.2f}",
                    'U_Statistic': f"{u_stat:.0f}",
                    'p-value': f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001",
                    'Effect_Size_r': f"{r:.3f}",
                    'Significant_Bonferroni': 'Yes' if p_val < bonferroni_threshold else 'No'
                })
            except:
                pass
    
    table3_overall = pd.DataFrame(overall_results)
    table3_posthoc = pd.DataFrame(posthoc_results)
    table3_pairwise = pd.DataFrame(pairwise_results)
    
    # Standardize tidy post-hoc with analysis type
    for r in tidy_posthoc_pairwise:
        r['analysis_type'] = 'Table 3: Overall'
    
    return table3_overall, table3_posthoc, table3_pairwise, tidy_posthoc_pairwise

"""
Table 4: 年齢・性別調整済みロジスティック回帰分析
"""
def create_table4_multivariate_analysis(df):
    results = []
    
    df_analysis = df.copy()
    df_analysis['sex_male'] = (df_analysis['sex'] == 'Male').astype(int)
    
    reference_category = 'Physical Abuse'
    comparison_categories = ['Neglect', 'Emotional Abuse', 'Sexual Abuse']
    
    outcomes = [
        ('has_caries', 'Caries Experience (DMFT>0)'),
        ('has_untreated_caries', 'Untreated Caries'),
    ]
    
    if 'gingivitis' in df_analysis.columns:
        df_analysis['gingivitis_binary'] = (df_analysis['gingivitis'] == 'Gingivitis').astype(int)
        outcomes.append(('gingivitis_binary', 'Gingivitis'))
    
    if 'needTOBEtreated' in df_analysis.columns:
        df_analysis['treatment_need'] = (df_analysis['needTOBEtreated'] == 'Treatment Required').astype(int)
        outcomes.append(('treatment_need', 'Treatment Need'))
    
    for outcome_var, outcome_label in outcomes:
        if outcome_var not in df_analysis.columns:
            continue
        
        for comparison in comparison_categories:
            df_model = df_analysis[df_analysis['abuse'].isin([reference_category, comparison])].copy()
            df_model = df_model[[outcome_var, 'age_year', 'sex_male', 'abuse']].dropna()
            
            if len(df_model) < 50:
                continue
            
            df_model['comparison'] = (df_model['abuse'] == comparison).astype(int)
            
            try:
                X = np.column_stack([
                    np.ones(len(df_model)),
                    df_model['age_year'].values,
                    df_model['sex_male'].values,
                    df_model['comparison'].values
                ])
                y = df_model[outcome_var].values
                
                result = simple_logistic_regression(X, y)
                
                odds_ratio = result['odds_ratios'][3]
                ci_lower = result['ci_lower'][3]
                ci_upper = result['ci_upper'][3]
                p_val = result['p_values'][3]
                
                results.append({
                    'Outcome': outcome_label,
                    'Comparison': f"{comparison} vs {reference_category}",
                    'Odds Ratio': f"{odds_ratio:.2f}",
                    '95% CI': f"({ci_lower:.2f}-{ci_upper:.2f})",
                    'p-value': f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001",
                    'Adjusted_for': 'Age, Sex'
                })
                
            except Exception as e:
                results.append({
                    'Outcome': outcome_label,
                    'Comparison': f"{comparison} vs {reference_category}",
                    'Odds Ratio': 'N/A',
                    '95% CI': 'N/A',
                    'p-value': 'N/A',
                    'Adjusted_for': 'Age, Sex'
                })
    
    table4 = pd.DataFrame(results)
    return table4

"""
Table 5: DMFT_Index analysis by life_stage and abuse type
-- Returns a DataFrame with descriptive statistics for each life_stage × abuse combination
"""  
def create_table5_dmft_by_lifestage_abuse(df):
    abuse_types = list(df['abuse'].cat.categories)
    life_stages = df['age_group'].dropna().unique()
    
    # Sort life_stages if they have a natural order
    life_stage_order = ['Early Childhood (2-6)', 
                        'Middle Childhood (7-12)', 
                        'Adolescence (13-18)']
    life_stages = [ls for ls in life_stage_order if ls in life_stages] + \
                  [ls for ls in life_stages if ls not in life_stage_order]
    
    results = []
    
    for life_stage in life_stages:
        df_stage = df[df['age_group'] == life_stage]
        
        # Kruskal-Wallis test across abuse types within this life_stage
        groups = [df_stage[df_stage['abuse'] == abuse]['DMFT_Index'].dropna() 
                  for abuse in abuse_types]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                h_stat, p_kw = kruskal(*groups)
                p_val_str = f"{p_kw:.4f}" if p_kw >= 0.0001 else "<0.0001"
            except:
                p_val_str = "N/A"
        else:
            p_val_str = "N/A"
        
        first_row = True
        for abuse in abuse_types:
            subset = df_stage[(df_stage['abuse'] == abuse)]['DMFT_Index'].dropna()
            
            if len(subset) == 0:
                continue
            
            row = {
                'Life_Stage': life_stage if first_row else '',
                'Abuse_Type': abuse,
                'N': len(subset),
                'Mean': f"{subset.mean():.2f}",
                'SD': f"{subset.std():.2f}",
                'Median': f"{subset.median():.1f}",
                '25%': f"{subset.quantile(0.25):.1f}",
                '75%': f"{subset.quantile(0.75):.1f}",
                'Min': f"{subset.min():.0f}",
                'Max': f"{subset.max():.0f}",
                'p-value (KW)': p_val_str if first_row else ''
            }
            results.append(row)
            first_row = False
    
    # Add overall summary by life stage (all abuse types combined)
    results.append({
        'Life_Stage': '=== OVERALL BY LIFE STAGE ===',
        'Abuse_Type': '(Combined)',
        'N': '---',
        'Mean': '---',
        'SD': '---',
        'Median': '---',
        '25%': '---',
        '75%': '---',
        'Min': '---',
        'Max': '---',
        'p-value (KW)': '---'
    })
    
    # Kruskal-Wallis test across life stages (all abuse types combined)
    life_stage_groups = [df[df['age_group'] == ls]['DMFT_Index'].dropna() 
                         for ls in life_stages]
    life_stage_groups = [g for g in life_stage_groups if len(g) > 0]
    
    if len(life_stage_groups) >= 2:
        try:
            h_stat, p_kw_lifestage = kruskal(*life_stage_groups)
            p_val_lifestage_str = f"{p_kw_lifestage:.4f}" if p_kw_lifestage >= 0.0001 else "<0.0001"
        except:
            p_val_lifestage_str = "N/A"
    else:
        p_val_lifestage_str = "N/A"
    
    first_lifestage = True
    for life_stage in life_stages:
        subset = df[df['age_group'] == life_stage]['DMFT_Index'].dropna()
        if len(subset) > 0:
            results.append({
                'Life_Stage': life_stage,
                'Abuse_Type': 'All abuse types',
                'N': len(subset),
                'Mean': f"{subset.mean():.2f}",
                'SD': f"{subset.std():.2f}",
                'Median': f"{subset.median():.1f}",
                '25%': f"{subset.quantile(0.25):.1f}",
                '75%': f"{subset.quantile(0.75):.1f}",
                'Min': f"{subset.min():.0f}",
                'Max': f"{subset.max():.0f}",
                'p-value (KW)': p_val_lifestage_str if first_lifestage else ''
            })
            first_lifestage = False
    
    tidy_posthoc = []
    # Add post-hoc for life stage strata
    for life_stage in life_stages:
        df_stage = df[df['age_group'] == life_stage]
        groups = [df_stage[df_stage['abuse'] == abuse]['DMFT_Index'].dropna() for abuse in abuse_types]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                _, p_kw = kruskal(*groups)
                if p_kw < 0.05:
                    dunn_adj, dunn_unadj = posthoc_dunn(df_stage, val_col='DMFT_Index', group_col='abuse', p_adjust='bonferroni')
                    for i, abuse1 in enumerate(abuse_types):
                        for abuse2 in abuse_types[i+1:]:
                            if abuse1 in dunn_adj.index and abuse2 in dunn_adj.columns:
                                tidy_posthoc.append({
                                    'analysis_type': f'Table 5: {life_stage}',
                                    'variable': 'DMFT_Index',
                                    'group1': abuse1,
                                    'group2': abuse2,
                                    'p_unadjusted': dunn_unadj.loc[abuse1, abuse2],
                                    'p_adjusted': dunn_adj.loc[abuse1, abuse2],
                                    'significant': dunn_adj.loc[abuse1, abuse2] < 0.05
                                })
            except: pass

    # Add post-hoc for overall life stage comparison
    if len(life_stage_groups) >= 2 and p_kw_lifestage < 0.05:
        try:
            dunn_adj, dunn_unadj = posthoc_dunn(df.dropna(subset=['age_group']), val_col='DMFT_Index', group_col='age_group', p_adjust='bonferroni')
            for i, ls1 in enumerate(life_stages):
                for ls2 in life_stages[i+1:]:
                    if ls1 in dunn_adj.index and ls2 in dunn_adj.columns:
                        tidy_posthoc.append({
                            'analysis_type': 'Table 5: Life Stage Overall',
                            'variable': 'DMFT_Index',
                            'group1': ls1,
                            'group2': ls2,
                            'p_unadjusted': dunn_unadj.loc[ls1, ls2],
                            'p_adjusted': dunn_adj.loc[ls1, ls2],
                            'significant': dunn_adj.loc[ls1, ls2] < 0.05
                        })
        except: pass

    # Add pairwise significance section to the table itself
    results.append({
        'Life_Stage': '=== POST-HOC pairwise (Dunn\'s) ===',
        'Abuse_Type': '(Only if KW p < 0.05)',
        'N': '', 'Mean': '', 'SD': '', 'Median': '', '25%': '', '75%': '', 'Min': '', 'Max': '', 'p-value (KW)': ''
    })
    
    for tp in tidy_posthoc:
        results.append({
            'Life_Stage': f"Post-hoc: {tp['analysis_type']}",
            'Abuse_Type': f"{tp['group1']} vs {tp['group2']}",
            'N': 'Significant' if tp['significant'] else 'n.s.',
            'Mean': f"p_adj={tp['p_adjusted']:.4f}",
            'SD': f"p_unadj={tp['p_unadjusted']:.4f}",
            'Median': '', '25%': '', '75%': '', 'Min': '', 'Max': '', 'p-value (KW)': ''
        })

    table5 = pd.DataFrame(results)
    return table5, tidy_posthoc

"""
Table 5.5: Caries Prevalence and Treatment Status Analysis
Calculates:
1. Percentage of children with caries (DMFT_Index > 0) - total and by abuse type
2. Percentage of children where f+F = DMFT_Index (fully treated caries)
3. Percentage of children where f+F = 0 (no filled teeth)
    
Also includes descriptive statistics for C0 variables:
- DMFT_C0 (total DMFT including C0)
- Perm_DMFT_C0 (permanent teeth DMFT including C0)
- Baby_DMFT_C0 (baby teeth dmft including C0)
    
Returns a DataFrame with these statistics broken down by abuse type  
"""
def create_table5_5_caries_prevalence_treatment(df):
    abuse_types = list(df['abuse'].cat.categories)
    
    results = []
    
    # ========== SECTION 1: Caries Prevalence ==========
    results.append({
        'Variable': '=== CARIES PREVALENCE ===',
        'Category': '',
        **{abuse: '' for abuse in abuse_types},
        'Total': '',
        'p-value': ''
    })
    
    # 1. Percentage of children with caries (DMFT_Index > 0)
    row_caries = {'Variable': 'Children with Caries', 'Category': 'DMFT_Index > 0'}
    
    for abuse in abuse_types:
        subset = df[df['abuse'] == abuse]
        n_total = len(subset)
        n_caries = (subset['DMFT_Index'] > 0).sum()
        pct = (n_caries / n_total * 100) if n_total > 0 else 0
        row_caries[abuse] = f"{n_caries}/{n_total} ({pct:.1f}%)"
    
    # Total
    n_total_all = len(df)
    n_caries_all = (df['DMFT_Index'] > 0).sum()
    pct_all = (n_caries_all / n_total_all * 100) if n_total_all > 0 else 0
    row_caries['Total'] = f"{n_caries_all}/{n_total_all} ({pct_all:.1f}%)"
    
    # Chi-square test for caries prevalence
    df['has_caries'] = (df['DMFT_Index'] > 0).astype(int)
    contingency = pd.crosstab(df['abuse'], df['has_caries'])
    try:
        chi2, p_val, _, _ = chi2_contingency(contingency)
        row_caries['p-value'] = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    except:
        row_caries['p-value'] = 'N/A'
    results.append(row_caries)
    
    # ========== SECTION 2: Treatment Status ==========
    results.append({
        'Variable': '=== TREATMENT STATUS ===',
        'Category': '',
        **{abuse: '' for abuse in abuse_types},
        'Total': '',
        'p-value': ''
    })
    
    # Calculate f+F for the dataset
    df['filled_total'] = df['Perm_F'] + df['Baby_f']
    
    # 2. Percentage of children where f+F = DMFT_Index (fully treated)
    row_fully_treated = {'Variable': 'Fully Treated Caries', 'Category': 'f+F = DMFT_Index'}
    
    # Only consider children with caries (DMFT_Index > 0)
    df_with_caries = df[df['DMFT_Index'] > 0]
    
    for abuse in abuse_types:
        subset = df_with_caries[df_with_caries['abuse'] == abuse]
        n_total = len(subset)
        n_fully_treated = (subset['filled_total'] == subset['DMFT_Index']).sum()
        pct = (n_fully_treated / n_total * 100) if n_total > 0 else 0
        row_fully_treated[abuse] = f"{n_fully_treated}/{n_total} ({pct:.1f}%)"
    
    n_total_caries = len(df_with_caries)
    n_fully_treated_all = (df_with_caries['filled_total'] == df_with_caries['DMFT_Index']).sum()
    pct_fully_treated = (n_fully_treated_all / n_total_caries * 100) if n_total_caries > 0 else 0
    row_fully_treated['Total'] = f"{n_fully_treated_all}/{n_total_caries} ({pct_fully_treated:.1f}%)"
    
    # Chi-square test for fully treated
    df_with_caries = df_with_caries.copy()
    df_with_caries['is_fully_treated'] = (df_with_caries['filled_total'] == df_with_caries['DMFT_Index']).astype(int)
    contingency_treated = pd.crosstab(df_with_caries['abuse'], df_with_caries['is_fully_treated'])
    try:
        chi2, p_val, _, _ = chi2_contingency(contingency_treated)
        row_fully_treated['p-value'] = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    except:
        row_fully_treated['p-value'] = 'N/A'
    results.append(row_fully_treated)
    
    # 3. Percentage of children where f+F = 0 (no filled teeth)
    row_no_filled = {'Variable': 'No Filled Teeth', 'Category': 'f+F = 0'}
    
    for abuse in abuse_types:
        subset = df[df['abuse'] == abuse]
        n_total = len(subset)
        n_no_filled = (subset['filled_total'] == 0).sum()
        pct = (n_no_filled / n_total * 100) if n_total > 0 else 0
        row_no_filled[abuse] = f"{n_no_filled}/{n_total} ({pct:.1f}%)"
    
    n_total_all = len(df)
    n_no_filled_all = (df['filled_total'] == 0).sum()
    pct_no_filled = (n_no_filled_all / n_total_all * 100) if n_total_all > 0 else 0
    row_no_filled['Total'] = f"{n_no_filled_all}/{n_total_all} ({pct_no_filled:.1f}%)"
    
    # Chi-square test for no filled teeth
    df['has_no_filled'] = (df['filled_total'] == 0).astype(int)
    contingency_nofilled = pd.crosstab(df['abuse'], df['has_no_filled'])
    try:
        chi2, p_val, _, _ = chi2_contingency(contingency_nofilled)
        row_no_filled['p-value'] = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
    except:
        row_no_filled['p-value'] = 'N/A'
    results.append(row_no_filled)
    
    # ========== SECTION 3: C0 Variables Analysis ==========
    results.append({
        'Variable': '=== DMFT WITH C0 (INCIPIENT CARIES) ===',
        'Category': '',
        **{abuse: '' for abuse in abuse_types},
        'Total': '',
        'p-value': ''
    })
    
    # C0 variables to analyze
    c0_vars = [
        ('DMFT_C0', 'Total DMFT + C0'),
        ('Perm_DMFT_C0', 'Permanent DMFT + C0'),
        ('Baby_DMFT_C0', 'Primary dmft + C0')
    ]
    
    for var_name, var_label in c0_vars:
        if var_name not in df.columns:
            continue
        
        # Header row
        row_header = {'Variable': var_label, 'Category': 'Mean ± SD'}
        for abuse in abuse_types:
            subset = df[df['abuse'] == abuse][var_name].dropna()
            if len(subset) > 0:
                row_header[abuse] = f"{subset.mean():.2f} ± {subset.std():.2f}"
            else:
                row_header[abuse] = 'N/A'
        
        total = df[var_name].dropna()
        if len(total) > 0:
            row_header['Total'] = f"{total.mean():.2f} ± {total.std():.2f}"
        else:
            row_header['Total'] = 'N/A'
        
        # Kruskal-Wallis test
        groups = [df[df['abuse'] == abuse][var_name].dropna() for abuse in abuse_types]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                _, p_val = kruskal(*groups)
                row_header['p-value'] = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
            except:
                row_header['p-value'] = 'N/A'
        else:
            row_header['p-value'] = 'N/A'
        results.append(row_header)
        
        # Median row
        row_median = {'Variable': '', 'Category': 'Median [IQR]'}
        for abuse in abuse_types:
            subset = df[df['abuse'] == abuse][var_name].dropna()
            if len(subset) > 0:
                q25, q50, q75 = subset.quantile([0.25, 0.5, 0.75])
                row_median[abuse] = f"{q50:.1f} [{q25:.1f}-{q75:.1f}]"
            else:
                row_median[abuse] = 'N/A'
        
        if len(total) > 0:
            q25, q50, q75 = total.quantile([0.25, 0.5, 0.75])
            row_median['Total'] = f"{q50:.1f} [{q25:.1f}-{q75:.1f}]"
        else:
            row_median['Total'] = 'N/A'
        row_median['p-value'] = ''
        results.append(row_median)
    
    # ========== SECTION 4: Prevalence with C0 ==========
    results.append({
        'Variable': '=== CARIES PREVALENCE (INCLUDING C0) ===',
        'Category': '',
        **{abuse: '' for abuse in abuse_types},
        'Total': '',
        'p-value': ''
    })
    
    # Percentage with DMFT_C0 > 0
    if 'DMFT_C0' in df.columns:
        row_c0_prev = {'Variable': 'Children with Caries (incl. C0)', 'Category': 'DMFT_C0 > 0'}
        
        for abuse in abuse_types:
            subset = df[df['abuse'] == abuse]
            n_total = len(subset)
            n_caries = (subset['DMFT_C0'] > 0).sum()
            pct = (n_caries / n_total * 100) if n_total > 0 else 0
            row_c0_prev[abuse] = f"{n_caries}/{n_total} ({pct:.1f}%)"
        
        n_total_all = len(df)
        n_caries_all = (df['DMFT_C0'] > 0).sum()
        pct_all = (n_caries_all / n_total_all * 100) if n_total_all > 0 else 0
        row_c0_prev['Total'] = f"{n_caries_all}/{n_total_all} ({pct_all:.1f}%)"
        
        # Chi-square test
        df['has_caries_c0'] = (df['DMFT_C0'] > 0).astype(int)
        contingency_c0 = pd.crosstab(df['abuse'], df['has_caries_c0'])
        try:
            chi2, p_val, _, _ = chi2_contingency(contingency_c0)
            row_c0_prev['p-value'] = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
        except:
            row_c0_prev['p-value'] = 'N/A'
        results.append(row_c0_prev)

    # ========== SECTION 5: Post-hoc Dunn's Tests for C0 variables ==========
    results.append({
        'Variable': '=== POST-HOC pairwise (Dunn\'s) ===',
        'Category': '(Only if KW p < 0.05)',
        **{abuse: '' for abuse in abuse_types},
        'Total': '',
        'p-value': ''
    })

    for var_name, var_label in c0_vars:
        if var_name not in df.columns:
            continue
        
        # Re-check KW p-value
        groups = [df[df['abuse'] == abuse][var_name].dropna() for abuse in abuse_types]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
            
        try:
            _, kw_p = kruskal(*groups)
            if kw_p >= 0.05:
                continue
                
            dunn_result = posthoc_dunn(df, val_col=var_name, group_col='abuse', p_adjust='bonferroni')
            
            for i, abuse1 in enumerate(abuse_types):
                for abuse2 in abuse_types[i+1:]:
                    if abuse1 in dunn_result.index and abuse2 in dunn_result.columns:
                        p_val = dunn_result.loc[abuse1, abuse2]
                        if p_val < 0.05:
                            results.append({
                                'Variable': f"Post-hoc: {var_label}",
                                'Category': f"{abuse1} vs {abuse2}",
                                **{abuse: (f"p={p_val:.4f}" if abuse == abuse1 or abuse == abuse2 else '') for abuse in abuse_types},
                                'Total': 'Significant',
                                'p-value': f"{p_val:.4f}"
                            })
        except:
            pass
    
    table5_5 = pd.DataFrame(results)
    
    tidy_posthoc = []
    for var_name, var_label in c0_vars:
        if var_name not in df.columns:
            continue
        groups = [df[df['abuse'] == abuse][var_name].dropna() for abuse in abuse_types]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue
        try:
            _, kw_p = kruskal(*groups)
            if kw_p < 0.05:
                dunn_adj, dunn_unadj = posthoc_dunn(df, val_col=var_name, group_col='abuse', p_adjust='bonferroni')
                for i, abuse1 in enumerate(abuse_types):
                    for abuse2 in abuse_types[i+1:]:
                        if abuse1 in dunn_adj.index and abuse2 in dunn_adj.columns:
                            tidy_posthoc.append({
                                'analysis_type': 'Table 5.5: C0 Stats',
                                'variable': var_name,
                                'group1': abuse1,
                                'group2': abuse2,
                                'p_unadjusted': dunn_unadj.loc[abuse1, abuse2],
                                'p_adjusted': dunn_adj.loc[abuse1, abuse2],
                                'significant': dunn_adj.loc[abuse1, abuse2] < 0.05
                            })
        except: pass

    return table5_5, tidy_posthoc

"""
Table 6: DMFT_Index analysis by dentition type and abuse type
Groups based on present teeth:
- primary_dentition: Present_Baby_Teeth == Present_Teeth (only baby teeth present)
- permanent_dentition: Present_Perm_Teeth == Present_Teeth (only permanent teeth present)
- mixed_dentition: Both baby and permanent teeth present
Returns a DataFrame with descriptive statistics for each dentition_type × abuse combination
"""
def create_table6_dmft_by_dentition_abuse(df):
    required_cols = ['DMFT_Index', 'Present_Teeth', 'Perm_total_teeth', 'Present_Perm_Teeth', 'abuse']
    for col in required_cols:
        if col not in df.columns:
            print(f"   ⚠ '{col}' column not found in data")
            return pd.DataFrame()
    
    # Create dentition type groups based on present teeth
    df_analysis = df.copy()
    
    def get_dentition_type(row):
        present_teeth = row['Present_Teeth'] if pd.notna(row['Present_Teeth']) else 0
        present_baby = row['Present_Baby_Teeth'] if pd.notna(row['Present_Baby_Teeth']) else 0
        present_perm = row['Present_Perm_Teeth'] if pd.notna(row['Present_Perm_Teeth']) else 0
        
        if present_teeth == 0:
            return 'No_Teeth'
        elif present_baby == present_teeth and present_perm == 0:
            return 'primary_dentition'
        elif present_perm == present_teeth and present_baby == 0:
            return 'permanent_dentition'
        else:
            return 'mixed_dentition'
    
    df_analysis['dentition_type'] = df_analysis.apply(get_dentition_type, axis=1)
    
    abuse_types = list(df['abuse'].cat.categories)
    dentition_order = ['primary_dentition', 'mixed_dentition', 'permanent_dentition']
    
    results = []
    
    for dent_type in dentition_order:
        df_dent = df_analysis[df_analysis['dentition_type'] == dent_type]
        
        if len(df_dent) == 0:
            continue
        
        # Kruskal-Wallis test across abuse types within this dentition type
        groups = [df_dent[df_dent['abuse'] == abuse]['DMFT_Index'].dropna() 
                  for abuse in abuse_types]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            try:
                h_stat, p_kw = kruskal(*groups)
                p_val_str = f"{p_kw:.4f}" if p_kw >= 0.0001 else "<0.0001"
            except:
                p_val_str = "N/A"
        else:
            p_val_str = "N/A"
        
        first_row = True
        for abuse in abuse_types:
            subset = df_dent[(df_dent['abuse'] == abuse)]['DMFT_Index'].dropna()
            
            if len(subset) == 0:
                continue
            
            row = {
                'Dentition_Type': dent_type if first_row else '',
                'Abuse_Type': abuse,
                'N': len(subset),
                'Mean': f"{subset.mean():.2f}",
                'SD': f"{subset.std():.2f}",
                'Median': f"{subset.median():.1f}",
                '25%': f"{subset.quantile(0.25):.1f}",
                '75%': f"{subset.quantile(0.75):.1f}",
                'Min': f"{subset.min():.0f}",
                'Max': f"{subset.max():.0f}",
                'p-value (KW)': p_val_str if first_row else ''
            }
            results.append(row)
            first_row = False
    
    # ========== SUMMARY SECTION ==========
    # Add separator
    results.append({
        'Dentition_Type': '=== SUMMARY BY DENTITION TYPE ===',
        'Abuse_Type': '',
        'N': '',
        'Mean': '',
        'SD': '',
        'Median': '',
        '25%': '',
        '75%': '',
        'Min': '',
        'Max': '',
        'p-value (KW)': ''
    })
    
    # Kruskal-Wallis test across dentition types (overall)
    dent_groups = [df_analysis[df_analysis['dentition_type'] == dt]['DMFT_Index'].dropna() 
                   for dt in dentition_order]
    dent_groups = [g for g in dent_groups if len(g) > 0]
    
    if len(dent_groups) >= 2:
        try:
            h_stat, p_kw_overall = kruskal(*dent_groups)
            p_kw_overall_str = f"{p_kw_overall:.4f}" if p_kw_overall >= 0.0001 else "<0.0001"
        except:
            p_kw_overall_str = "N/A"
    else:
        p_kw_overall_str = "N/A"
    
    # Add descriptive stats for each dentition type (all abuse types combined)
    first_summary = True
    for dent_type in dentition_order:
        subset = df_analysis[df_analysis['dentition_type'] == dent_type]['DMFT_Index'].dropna()
        if len(subset) == 0:
            continue
        
        n_total = len(subset)
        pct = (n_total / len(df_analysis) * 100) if len(df_analysis) > 0 else 0
        
        results.append({
            'Dentition_Type': dent_type,
            'Abuse_Type': f'All abuse types (n={n_total}, {pct:.1f}%)',
            'N': n_total,
            'Mean': f"{subset.mean():.2f}",
            'SD': f"{subset.std():.2f}",
            'Median': f"{subset.median():.1f}",
            '25%': f"{subset.quantile(0.25):.1f}",
            '75%': f"{subset.quantile(0.75):.1f}",
            'Min': f"{subset.min():.0f}",
            'Max': f"{subset.max():.0f}",
            'p-value (KW)': f"Overall KW: {p_kw_overall_str}" if first_summary else ''
        })
        first_summary = False
    
    # Pairwise Mann-Whitney U tests between dentition types
    results.append({
        'Dentition_Type': '--- Pairwise Comparisons (Dunn\'s Test with Bonferroni) ---',
        'Abuse_Type': '',
        'N': '',
        'Mean': '',
        'SD': '',
        'Median': '',
        '25%': '',
        '75%': '',
        'Min': '',
        'Max': '',
        'p-value (KW)': ''
    })
    
    # Perform Dunn's test for pairwise comparisons
    try:
        dunn_results = sp.posthoc_dunn(
            df_analysis[df_analysis['dentition_type'].isin(dentition_order)],
            val_col='DMFT_Index',
            group_col='dentition_type',
            p_adjust='bonferroni'
        )
        
        dentition_pairs = list(itertools.combinations(dentition_order, 2))
        
        for dent1, dent2 in dentition_pairs:
            if dent1 in dunn_results.index and dent2 in dunn_results.columns:
                p_val = dunn_results.loc[dent1, dent2]
                
                # Get sample sizes and medians
                data1 = df_analysis[df_analysis['dentition_type'] == dent1]['DMFT_Index'].dropna()
                data2 = df_analysis[df_analysis['dentition_type'] == dent2]['DMFT_Index'].dropna()
                
                # Significance stars
                if p_val <= 0.001:
                    sig = '***'
                elif p_val <= 0.01:
                    sig = '**'
                elif p_val <= 0.05:
                    sig = '*'
                else:
                    sig = ''
                
                p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"
                
                results.append({
                    'Dentition_Type': f'{dent1} vs {dent2}',
                    'Abuse_Type': f'Dunn\'s test{sig}',
                    'N': f'{len(data1)} vs {len(data2)}',
                    'Mean': '',
                    'SD': '',
                    'Median': f'{data1.median():.1f} vs {data2.median():.1f}',
                    '25%': '',
                    '75%': '',
                    'Min': '',
                    'Max': '',
                    'p-value (KW)': f'p={p_str}{sig}'
                })
    except Exception as e:
        print(f"   ⚠ Dunn's test failed: {e}")
    
    table6 = pd.DataFrame(results)
    return table6

"""
Perform pairwise Mann-Whitney U tests between all groups.
Parameters:
- df: DataFrame with data
- var_name: Name of the continuous variable to compare (e.g., 'DMFT_Index')
- group_col: Name of the grouping column (e.g., 'abuse')
- p_adjust: Method for p-value adjustment ('bonferroni' or None)
Returns:
- DataFrame with pairwise comparison results
"""
def pairwise_mannwhitney(df, var_name, group_col='abuse', p_adjust='bonferroni'):
    # Get unique groups
    if df[group_col].dtype.name == 'category':
        groups = df[group_col].cat.categories.tolist()
    else:
        groups = sorted(df[group_col].dropna().unique())
    
    # Generate all pairwise combinations
    pairs = list(itertools.combinations(groups, 2))
    n_comparisons = len(pairs)
    
    results = []
    
    for group1, group2 in pairs:
        # Get data for each group
        data1 = df[df[group_col] == group1][var_name].dropna()
        data2 = df[df[group_col] == group2][var_name].dropna()
        
        if len(data1) == 0 or len(data2) == 0:
            continue
        
        # Perform Mann-Whitney U test
        u_stat, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation r)
        n1, n2 = len(data1), len(data2)
        r = 1 - (2 * u_stat) / (n1 * n2)
        
        # Adjust p-value if requested
        if p_adjust == 'bonferroni':
            p_adjusted = min(p_val * n_comparisons, 1.0)
        else:
            p_adjusted = p_val
        
        # Format p-value
        if p_adjusted < 0.0001:
            p_str = '<0.0001'
        else:
            p_str = f'{p_adjusted:.4f}'
        
        # Significance stars
        if p_adjusted <= 0.001:
            sig = '***'
        elif p_adjusted <= 0.01:
            sig = '**'
        elif p_adjusted <= 0.05:
            sig = '*'
        else:
            sig = ''
        
        results.append({
            'Group1': group1,
            'Group2': group2,
            'N1': n1,
            'N2': n2,
            'Median1': f'{data1.median():.2f}',
            'Median2': f'{data2.median():.2f}',
            'U_Statistic': f'{u_stat:.0f}',
            'p-value_raw': f'{p_val:.4f}' if p_val >= 0.0001 else '<0.0001',
            'p-value_adjusted': p_str,
            'Effect_Size_r': f'{r:.3f}',
            'Significance': sig
        })
    
    return pd.DataFrame(results)

"""
Analyze DMFT by dentition type with pairwise Mann-Whitney U tests.
Creates 3 groups based on present teeth:
- primary_dentition: Present_Baby_Teeth == Present_Teeth (only baby teeth present)
- permanent_dentition: Present_Perm_Teeth == Present_Teeth (only permanent teeth present)
- mixed_dentition: Both baby and permanent teeth present
Then performs pairwise Mann-Whitney U tests between abuse types within each group.
Returns:
- DataFrame with pairwise comparison results for each dentition type
"""
def analyze_dmft_by_dentition_with_pairwise(df):
    
    required_cols = ['DMFT_Index', 'Present_Teeth', 'Present_Baby_Teeth', 'Present_Perm_Teeth', 'abuse']
    for col in required_cols:
        if col not in df.columns:
            print(f"   ⚠ '{col}' column not found in data")
            return pd.DataFrame()
    
    # Create dentition type column based on present teeth
    def get_dentition_type(row):
        present_teeth = row['Present_Teeth'] if pd.notna(row['Present_Teeth']) else 0
        present_baby = row['Present_Baby_Teeth'] if pd.notna(row['Present_Baby_Teeth']) else 0
        present_perm = row['Present_Perm_Teeth'] if pd.notna(row['Present_Perm_Teeth']) else 0
        
        if present_teeth == 0:
            return 'No_Teeth'
        elif present_baby == present_teeth and present_perm == 0:
            return 'primary_dentition'
        elif present_perm == present_teeth and present_baby == 0:
            return 'permanent_dentition'
        else:
            return 'mixed_dentition'
    
    df_analysis = df.copy()
    df_analysis['dentition_type'] = df_analysis.apply(get_dentition_type, axis=1)
    
    dentition_order = ['primary_dentition', 'mixed_dentition', 'permanent_dentition']
    all_pairwise_results = []
    
    for dent_type in dentition_order:
        df_subset = df_analysis[df_analysis['dentition_type'] == dent_type]
        
        if len(df_subset) < 10:  # Skip if too few samples
            print(f"   ⚠ Skipping {dent_type}: only {len(df_subset)} samples")
            continue
        
        # Perform pairwise Mann-Whitney U tests
        pairwise_df = pairwise_mannwhitney(df_subset, 'DMFT_Index', 'abuse', 'bonferroni')
        if not pairwise_df.empty:
            pairwise_df.insert(0, 'Dentition_Type', dent_type)
            all_pairwise_results.append(pairwise_df)
    
    if all_pairwise_results:
        return pd.concat(all_pairwise_results, ignore_index=True)
    else:
        return pd.DataFrame()

"""
95% CI文字列からlowerとupperを抽出
"""
def parse_ci(ci_str):
    import re
    match = re.search(r'\(([\d.]+)-([\d.]+)\)', ci_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan

"""
縦型フォレストプロット（ロジスティック回帰結果の可視化）    
Parameters:
-----------
df_logistic : DataFrame
    ロジスティック回帰の結果（Table4）
df_original : DataFrame
    元のデータフレーム（サンプルサイズ計算用）
output_dir : str
    出力ディレクトリ
timestamp : str
    タイムスタンプ
figsize : tuple
    図のサイズ
"""
def create_forest_plot_vertical(df_logistic, df_original, output_dir, timestamp, figsize=(10, 10)):
    import matplotlib.patches as mpatches
    
    df = df_logistic.copy()
    
    # N/Aを含む行を除外
    df = df[df['Odds Ratio'] != 'N/A']
    
    if df.empty:
        print("   ⚠ No valid data for forest plot")
        return
    
    # オッズ比を数値に変換
    df['OR'] = pd.to_numeric(df['Odds Ratio'], errors='coerce')
    
    # CIの解析
    df[['CI_lower', 'CI_upper']] = df['95% CI'].apply(
        lambda x: pd.Series(parse_ci(x))
    )
    
    # p値を数値に変換するヘルパー関数
    def parse_p_value(x):
        """Convert p-value string to numeric value."""
        if pd.isna(x):
            return np.nan
        if x == '<0.0001':
            return 0.0001
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan
    
    # p値をフォーマットするヘルパー関数
    def format_p_value(p):
        """
        Format p-value with significance stars.
        *** p ≤ 0.001
        **  p ≤ 0.01
        *   p ≤ 0.05
        """
        if pd.isna(p):
            return ''
        if p <= 0.001:
            return '≤0.001***'
        elif p <= 0.01:
            return f'{p:.3f}**'
        elif p <= 0.05:
            return f'{p:.2f}*'
        else:
            return f'{p:.2f}'
    
    # p値を数値に変換
    df['p_numeric'] = df['p-value'].apply(parse_p_value)
    
    # p値をフォーマット（有意性の星印付き）
    df['p_formatted'] = df['p_numeric'].apply(format_p_value)
    
    # 有意性のフラグ
    df['significant'] = df['p_numeric'] < 0.05
    
    # アウトカムの順序
    outcome_order = df['Outcome'].unique()
    
    # 比較群の色（カラーブラインドフレンドリー: Wong palette）
    # https://www.nature.com/articles/nmeth.1618
    comparison_colors = {
        'Neglect vs Physical Abuse': '#E69F00',           # Orange
        'Emotional Abuse vs Physical Abuse': '#56B4E9',   # Sky Blue  
        'Sexual Abuse vs Physical Abuse': '#009E73'       # Bluish Green
    }
    
    # サンプルサイズを計算
    abuse_sample_sizes = {}
    for abuse in ['Physical Abuse', 'Neglect', 'Emotional Abuse', 'Sexual Abuse']:
        abuse_sample_sizes[abuse] = len(df_original[df_original['abuse'] == abuse])
    
    # 比較群のサンプルサイズラベル
    comparison_n = {
        'Neglect': f"Neglect\n(n={abuse_sample_sizes['Neglect']})",
        'Emotional Abuse': f"Emotional Abuse\n(n={abuse_sample_sizes['Emotional Abuse']})",
        'Sexual Abuse': f"Sexual Abuse\n(n={abuse_sample_sizes['Sexual Abuse']})"
    }
    
    # プロット作成
    fig, ax = plt.subplots(figsize=figsize)
    
    # Y軸の位置を計算
    y_positions = []
    y_labels = []
    y_pos = 0
    outcome_positions = {}
    
    for outcome in outcome_order:
        outcome_data = df[df['Outcome'] == outcome]
        outcome_positions[outcome] = []
        
        for _, row in outcome_data.iterrows():
            y_positions.append(y_pos)
            # サンプルサイズ付きラベル
            comparison_short = row['Comparison'].replace(' vs Physical Abuse', '')
            label_with_n = comparison_n.get(comparison_short, comparison_short)
            y_labels.append(label_with_n)
            outcome_positions[outcome].append(y_pos)
            y_pos += 1
        
        y_pos += 0.5  # アウトカム間のスペース
    
    # 参照線（OR=1）
    ax.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    
    # エラーバーとポイントのプロット
    for i, (idx, row) in enumerate(df.iterrows()):
        y = y_positions[i]
        or_val = row['OR']
        ci_low = row['CI_lower']
        ci_up = row['CI_upper']
        
        # 色の選択
        color = comparison_colors.get(row['Comparison'], '#3498db')
        
        # マーカーサイズ（有意な場合は大きく）
        marker_size = 150 if row['significant'] else 100
        marker = 's' if row['significant'] else 'o'
        
        # エラーバー
        ax.errorbar(or_val, y, xerr=[[or_val - ci_low], [ci_up - or_val]], 
                   fmt='none', color=color, capsize=4, capthick=2, linewidth=2, zorder=2)
        
        # ポイント
        ax.scatter(or_val, y, s=marker_size, c=color, marker=marker, 
                  edgecolors='white', linewidth=1.5, zorder=3)
        
        # OR値とCI、p値のテキスト
        text_x = max(ci_up + 0.1, 2.5)
        or_text = f"{or_val:.2f} ({ci_low:.2f}-{ci_up:.2f})"
        # Use 'p' instead of 'p=' when formatted value starts with '≤'
        p_formatted = row['p_formatted']
        p_text = f"p{p_formatted}" if p_formatted.startswith('≤') else f"p={p_formatted}"
        
        ax.annotate(or_text, xy=(text_x, y), fontsize=9, va='center')
        ax.annotate(p_text, xy=(text_x + 1.2, y), fontsize=8, va='center', 
                   color='red' if row['significant'] else 'gray')
    
    # アウトカムラベル（左側に追加）
    for outcome in outcome_order:
        positions = outcome_positions[outcome]
        if positions:
            mid_y = np.mean(positions)
            ax.annotate(outcome, xy=(-0.3, mid_y), fontsize=11, fontweight='bold',
                       va='center', ha='right', 
                       xycoords=('axes fraction', 'data'))
            
            # アウトカム間の区切り線
            if outcome != list(outcome_order)[-1]:
                max_y = max(positions)
                ax.axhline(y=max_y + 0.25, color='lightgray', linestyle='-', 
                          linewidth=0.5, alpha=0.7)
    
    # Y軸の設定
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.invert_yaxis()  # 上から下へ
    
    # X軸の設定
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 4.5)
    
    # タイトル（参照群のサンプルサイズも表示）
    ax.set_title(f'Adjusted Odds Ratios by Abuse Type\n(Reference: Physical Abuse n={abuse_sample_sizes["Physical Abuse"]}, Adjusted for Age and Sex)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # # 凡例
    # legend_elements = [
    #     mpatches.Patch(color=comparison_colors['Neglect vs Physical Abuse'], 
    #                   label='Neglect'),
    #     mpatches.Patch(color=comparison_colors['Emotional Abuse vs Physical Abuse'], 
    #                   label='Emotional Abuse'),
    #     mpatches.Patch(color=comparison_colors['Sexual Abuse vs Physical Abuse'], 
    #                   label='Sexual Abuse'),
    # ]
    # ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)
    
    # グリッド
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    
    # 枠線の調整
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存
    output_path_png = f'{output_dir}figure_forest_plot_{timestamp}.png'
    output_path_tiff = f'{output_dir}figure_forest_plot_{timestamp}.tiff'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_tiff, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Forest plot saved: figure_forest_plot_{timestamp}.png")
    print(f"   ✓ Forest plot saved: figure_forest_plot_{timestamp}.tiff")


"""
論文用の図を作成
"""
def create_visualizations(df, output_dir):
    abuse_order = ["Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse"]
    # カラーブラインドフレンドリー配色（Wong palette）
    colors = ['#0072B2', '#E69F00', '#56B4E9', '#009E73']  # Blue, Orange, Sky Blue, Bluish Green
    
    df_plot = df[df['abuse'].isin(abuse_order)].copy()
    
    # サンプルサイズを計算
    sample_sizes = {abuse: len(df_plot[df_plot['abuse'] == abuse]) for abuse in abuse_order}
    
    # サンプルサイズ付きラベルを作成
    abuse_labels_with_n = [f"{abuse}\n(n={sample_sizes[abuse]})" for abuse in abuse_order]
    
    if 'DMFT_Index' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(x='abuse', y='DMFT_Index', data=df_plot, 
                    order=abuse_order, palette=colors, ax=ax)
        
        ax.set_xlabel('Abuse Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('DMFT Index', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of DMFT Index by Abuse Type', fontsize=16, fontweight='bold')
        ax.set_xticklabels(abuse_labels_with_n, fontsize=10)
        # ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}figure1_dmft_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.violinplot(x='abuse', y='age_year', data=df_plot,
                   order=abuse_order, palette=colors, ax=ax)
    
    ax.set_xlabel('Abuse Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Age (years)', fontsize=14, fontweight='bold')
    ax.set_title('Age Distribution by Abuse Type', fontsize=16, fontweight='bold')
    ax.set_xticklabels(abuse_labels_with_n, fontsize=10)
    # ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}figure2_age_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    oral_vars = ['DMFT_Index', 'UTN_Score', 'Care_Index', 'Healthy_Rate', 'C0_Count', 'Trauma_Count']
    oral_vars_available = [v for v in oral_vars if v in df.columns]
    
    if len(oral_vars_available) > 0:
        mean_by_abuse = df_plot.groupby('abuse', observed=True)[oral_vars_available].mean()
        mean_by_abuse = mean_by_abuse.reindex(abuse_order)
        
        mean_normalized = (mean_by_abuse - mean_by_abuse.mean()) / mean_by_abuse.std()
        
        # ヒートマップ用のサンプルサイズ付きラベル
        heatmap_labels = [f"{abuse}\n(n={sample_sizes[abuse]})" for abuse in abuse_order]
        mean_normalized.index = heatmap_labels
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(mean_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn_r',
                    center=0, ax=ax, cbar_kws={'label': 'Z-score'})
        
        ax.set_xlabel('Abuse Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Oral Health Indicator', fontsize=14, fontweight='bold')
        ax.set_title('Standardized Oral Health Indicators by Abuse Type', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}figure3_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    cat_vars = [
        ('gingivitis', 'Gingivitis'),
        ('needTOBEtreated', 'Treatment Need'),
        ('OralCleanStatus', 'Oral Hygiene Status')
    ]
    
    for var_name, var_label in cat_vars:
        if var_name not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        df_valid = df_plot.dropna(subset=[var_name])
        
        # 各カテゴリのサンプルサイズを計算（有効データのみ）
        cat_sample_sizes = {abuse: len(df_valid[df_valid['abuse'] == abuse]) for abuse in abuse_order}
        cat_labels_with_n = [f"{abuse}\n(n={cat_sample_sizes[abuse]})" for abuse in abuse_order]
        
        crosstab = pd.crosstab(df_valid['abuse'], df_valid[var_name], normalize='index') * 100
        crosstab = crosstab.reindex(abuse_order)
        
        crosstab.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Abuse Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{var_label} Distribution by Abuse Type', fontsize=16, fontweight='bold')
        ax.set_xticklabels(cat_labels_with_n, fontsize=10, 
                        #    rotation=15, 
                           ha='right')
        # ax.legend(title=var_label, bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}figure_{var_name}_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Figures saved to: {output_dir}")

"""
Kruskal-Wallis検定後のDunn検定を行い、有意差があるペアをBoxplot上に描画する関数    
Parameters:
- df: DataFrame
- var_name: 分析対象の連続変数名 (例: 'DMFT_Index')
- group_col: 群分けの変数名 (例: 'abuse')
- title: グラフのタイトル (Noneの場合は変数名を使用)
- p_adjust: Dunn検定の補正方法 ('bonferroni', 'holm' など)
"""
def plot_boxplot_with_dunn(df, var_name, group_col='abuse', title=None, output_dir=OUTPUT_DIR,
                           p_adjust='bonferroni', palette='Set2', yaxis_name=None):
    # データをドロップ（欠損値除去）
    data = df[[group_col, var_name]].dropna()
    
    # カテゴリの順序を取得（category型の場合）
    if data[group_col].dtype.name == 'category':
        categories = data[group_col].cat.categories.tolist()
    else:
        categories = sorted(data[group_col].unique())
    
    # 1. Dunn's Testの実施
    try:
        # scikit-posthocsのposthoc_dunnを使用
        dunn_results = sp.posthoc_dunn(data, val_col=var_name, group_col=group_col, p_adjust=p_adjust)
    except Exception as e:
        print(f"Dunn検定エラー: {e}")
        return

    # 2. Boxplotの描画
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(x=group_col, y=var_name, data=data, order=categories, palette=palette, fill=False, legend=False, linewidth=2)
    sns.stripplot(x=group_col, y=var_name, data=data, order=categories, jitter=True, alpha=0.5, size=5, color=".3")

    # 3. 有意差ラインの描画準備
    significant_combinations = []
    
    # Dunn検定の結果から有意なペアを抽出
    # マトリックス形式の結果をループ処理
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i < j: # 重複を避けるため上三角のみ
                try:
                    p_val = dunn_results.loc[cat1, cat2]
                    if p_val < 0.05:
                        significant_combinations.append(((cat1, cat2), p_val))
                except KeyError:
                    continue

    # 有意差ラインを描画するための高さ設定
    y_max = data[var_name].max()
    y_range = y_max - data[var_name].min()
    h_step = y_range * 0.1  # ライン間の高さの間隔（データの10%分）
    y_start = y_max + (y_range * 0.05) # 最初のラインの開始位置
    
    # 有意なペアごとにラインを描画
    # 見やすくするために、距離が近いペアから順に書くなどの工夫も可能ですが、今回は単純ループ
    for idx, ((cat1, cat2), p_val) in enumerate(significant_combinations):
        # x座標の取得
        x1 = categories.index(cat1)
        x2 = categories.index(cat2)
        
        # ラインのy座標
        y = y_start + (idx * h_step)
        h = y_range * 0.02 # フックの高さ
        
        # ラインとフックの描画
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        
        # p値のテキスト表示
        # p値の表記形式（アスタリスクにするか数値にするか）
        if p_val < 0.001:
            label = "***"
        elif p_val < 0.01:
            label = "**"
        elif p_val < 0.05:
            label = "*"
        else:
            label = "ns"
            
        # 数値で表示したい場合はこちらを使用: label = f"p={p_val:.3f}"
        
        if np.isfinite(x1) and np.isfinite(x2) and np.isfinite(y) and np.isfinite(h):
            plt.text((x1+x2)*.5, y+h, label, ha='center', va='bottom', color='k', fontsize=10)

    # グラフの体裁を整える
    # ラインを描いた分だけY軸の上限を広げる
    if len(significant_combinations) > 0:
        plt.ylim(top=y_start + (len(significant_combinations) * h_step) + h_step)
    
    # abuse_order を使ってループすることで位置ズレを防ぎます
    for i, abuse_type in enumerate(categories):
        subset = data[data[group_col] == abuse_type][var_name]
        mean_val = subset.mean()
        
        # Draw mean line
        ax.hlines(mean_val, i - 0.4, i + 0.4, colors='red', linestyles='--', linewidth=2.5, 
                label='Mean' if i == 0 else '', zorder=10)
        
        # Add mean value text
        if pd.notna(mean_val) and np.isfinite(mean_val):
            ax.text(i + 0.45, mean_val, f'{mean_val:.2f}', fontsize=14, color='red', 
                    va='center', fontweight='bold')


    # 4. Add sample sizes to x-axis labels
    # 現在のラベルを取得し、n数を追記
    # 順序固定しているため、ラベル取得も安全に行えます
    labels = [item.get_text() for item in ax.get_xticklabels()]
    new_labels = [f'{label}\n(n={len(data[data[group_col] == label])})' for label in labels]
    ax.set_xticklabels(new_labels, fontsize=18)

    # その他設定
    labelSize= 18
    ax.tick_params(axis='y', labelsize=labelSize)
    ax.tick_params(axis='x', labelsize=labelSize)

    FontSize = 23
    plt.xlabel('Maltreatment Category', fontsize=FontSize, fontweight='bold')
    plt.ylabel(yaxis_name if yaxis_name else var_name, fontsize=FontSize, fontweight='bold')

    plt.title(title if title else f'{var_name} by Maltreatment Type (Dunn\'s Test)')
    plt.tight_layout()
    # plt.show()
    # 保存
    output_path_png = f'{output_dir}pairwise_results_{var_name}_{timestamp}.png'
    output_path_tiff = f'{output_dir}pairwise_results_{var_name}_{timestamp}.tiff'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_tiff, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Pairwise results saved: pairwise_results_{var_name}_{timestamp}.png")
    print(f"   ✓ Pairwise results saved: pairwise_results_{var_name}_{timestamp}.tiff")

"""
Boxplot of DMFT_Index by dentition type with Dunn's test pairwise comparisons.
    
Creates 3 groups based on present teeth:
- primary_dentition: Only baby teeth present
- mixed_dentition: Both baby and permanent teeth present
- permanent_dentition: Only permanent teeth present
    
Parameters:
- df: DataFrame with dental data
- output_dir: Output directory for saving plots
- p_adjust: P-value adjustment method for Dunn's test
- palette: Color palette for boxplots
"""
def plot_boxplot_by_dentition_type(df, output_dir=OUTPUT_DIR, p_adjust='bonferroni', palette='Set2'):
    
    required_cols = ['DMFT_Index', 'total_teeth', 'Baby_total_teeth', 'Perm_total_teeth']
    for col in required_cols:
        if col not in df.columns:
            print(f"   ⚠ '{col}' column not found in data")
            return
    
    # Create dentition type column
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
    
    df_analysis = df.copy()
    df_analysis['dentition_type'] = df_analysis.apply(get_dentition_type, axis=1)
    
    # Filter to only include the 3 dentition types
    dentition_order = ['primary_dentition', 'mixed_dentition', 'permanent_dentition']
    data = df_analysis[df_analysis['dentition_type'].isin(dentition_order)][['dentition_type', 'DMFT_Index']].dropna()
    
    if len(data) == 0:
        print("   ⚠ No data available for plotting")
        return
    
    # 1. Dunn's Test
    try:
        dunn_results = sp.posthoc_dunn(data, val_col='DMFT_Index', group_col='dentition_type', p_adjust=p_adjust)
    except Exception as e:
        print(f"   ⚠ Dunn's test error: {e}")
        return
    
    # 2. Create boxplot
    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='dentition_type', y='DMFT_Index', data=data, order=dentition_order, 
                     palette=palette, fill=False, legend=False, linewidth=2)
    sns.stripplot(x='dentition_type', y='DMFT_Index', data=data, order=dentition_order, 
                  jitter=True, alpha=0.5, size=5, color=".3")
    
    # 3. Extract significant pairs
    significant_combinations = []
    for i, cat1 in enumerate(dentition_order):
        for j, cat2 in enumerate(dentition_order):
            if i < j:
                try:
                    p_val = dunn_results.loc[cat1, cat2]
                    if p_val < 0.05:
                        significant_combinations.append(((cat1, cat2), p_val))
                except KeyError:
                    continue
    
    # 4. Draw significance lines
    y_max = data['DMFT_Index'].max()
    y_range = y_max - data['DMFT_Index'].min()
    h_step = y_range * 0.1
    y_start = y_max + (y_range * 0.05)
    
    for idx, ((cat1, cat2), p_val) in enumerate(significant_combinations):
        x1 = dentition_order.index(cat1)
        x2 = dentition_order.index(cat2)
        
        y = y_start + (idx * h_step)
        h = y_range * 0.02
        
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
        
        # Significance stars
        if p_val < 0.001:
            label = "***"
        elif p_val < 0.01:
            label = "**"
        elif p_val < 0.05:
            label = "*"
        else:
            label = "ns"
        
        if np.isfinite(x1) and np.isfinite(x2) and np.isfinite(y) and np.isfinite(h):
            plt.text((x1+x2)*.5, y+h, label, ha='center', va='bottom', color='k', fontsize=12)
    
    # Adjust y-axis limit
    if len(significant_combinations) > 0:
        plt.ylim(top=y_start + (len(significant_combinations) * h_step) + h_step)
    
    # 5. Add mean lines
    for i, dent_type in enumerate(dentition_order):
        subset = data[data['dentition_type'] == dent_type]['DMFT_Index']
        mean_val = subset.mean()
        
        ax.hlines(mean_val, i - 0.4, i + 0.4, colors='red', linestyles='--', linewidth=2.5, 
                  label='Mean' if i == 0 else '', zorder=10)
        
        if pd.notna(mean_val) and np.isfinite(mean_val):
            ax.text(i + 0.45, mean_val, f'{mean_val:.2f}', fontsize=14, color='red', 
                    va='center', fontweight='bold')
    
    # 6. Update x-axis labels with sample sizes
    dentition_labels = {
        'primary_dentition': 'Primary\nDentition',
        'mixed_dentition': 'Mixed\nDentition',
        'permanent_dentition': 'Permanent\nDentition'
    }
    new_labels = [f"{dentition_labels[dt]}\n(n={len(data[data['dentition_type'] == dt])})" 
                  for dt in dentition_order]
    ax.set_xticklabels(new_labels, fontsize=16)
    
    # 7. Styling
    labelSize = 18
    ax.tick_params(axis='y', labelsize=labelSize)
    ax.tick_params(axis='x', labelsize=labelSize)
    
    FontSize = 23
    plt.xlabel('Dentition Type', fontsize=FontSize, fontweight='bold')
    plt.ylabel('DMFT Index', fontsize=FontSize, fontweight='bold')
    plt.title("DMFT Index by Dentition Type (Dunn's Test with Bonferroni)", fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path_png = f'{output_dir}pairwise_results_dentition_type_{timestamp}.png'
    output_path_tiff = f'{output_dir}pairwise_results_dentition_type_{timestamp}.tiff'
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_tiff, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✓ Dentition type plot saved: pairwise_results_dentition_type_{timestamp}.png")
    print(f"   ✓ Dentition type plot saved: pairwise_results_dentition_type_{timestamp}.tiff")

"""
解析結果のサマリーレポートを生成
"""
def generate_summary_report(df, table3_overall, output_dir, timestamp):
    abuse_types = df['abuse'].cat.categories
    
    report = []
    report.append("=" * 70)
    report.append("SUMMARY REPORT: Oral Health by Abuse Type")
    report.append("=" * 70)
    report.append("")
    
    report.append("1. SAMPLE SIZES")
    report.append("-" * 40)
    for abuse in abuse_types:
        n = len(df[df['abuse'] == abuse])
        report.append(f"   {abuse}: n = {n}")
    report.append(f"   Total: n = {len(df)}")
    report.append("")
    
    report.append("2. KEY FINDINGS (Significant at p < 0.05)")
    report.append("-" * 40)
    
    if not table3_overall.empty:
        sig_results = table3_overall[table3_overall['Significant'] == 'Yes']
        if len(sig_results) > 0:
            for _, row in sig_results.iterrows():
                report.append(f"   • {row['Variable']}: p = {row['p-value']}")
        else:
            report.append("   No significant differences found.")
    report.append("")
    
    report.append("3. DMFT INDEX BY ABUSE TYPE")
    report.append("-" * 40)
    if 'DMFT_Index' in df.columns:
        for abuse in abuse_types:
            subset = df[df['abuse'] == abuse]['DMFT_Index'].dropna()
            if len(subset) > 0:
                report.append(f"   {abuse}:")
                report.append(f"      Mean ± SD: {subset.mean():.2f} ± {subset.std():.2f}")
                report.append(f"      Median [IQR]: {subset.median():.1f} [{subset.quantile(0.25):.1f}-{subset.quantile(0.75):.1f}]")
    report.append("")
    
    report_text = "\n".join(report)
    with open(f'{output_dir}summary_report_{timestamp}.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"   ✓ Saved: summary_report_{timestamp}.txt")

"""
Simple Logistic Regression Helper
"""
def simple_logistic_regression(X, y):
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0)
        
        params = result.params
        conf_int = result.conf_int()
        p_values = result.pvalues
        
        odds_ratios = np.exp(params)
        ci_lower = np.exp(conf_int[:, 0])
        ci_upper = np.exp(conf_int[:, 1])
        
        return {
            'odds_ratios': odds_ratios,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_values': p_values
        }
    except Exception as e:
        raise e


