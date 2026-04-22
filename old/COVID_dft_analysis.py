"""
Created: 2025-11-30
Last Edited: 2025-11-30
Summary: Calculates DMFT indices and performs Kruskal-Wallis and Dunn's tests on DFT index across Pre-COVID/COVID/After COVID periods.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import scikit_posthocs as sp
import numpy as np

# Load data
data_path = "/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data/data_noNA_OnlyAbuse.csv"
data0 = pd.read_csv(data_path)

# Define columns
perm_cols = [
    'U17', 'U16', 'U15', 'U14', 'U13', 'U12', 'U11', 'U21', 'U22', 'U23', 'U24', 'U25', 'U26', 'U27', 
    'L37', 'L36', 'L35', 'L34', 'L33', 'L32', 'L31', 'L41', 'L42', 'L43', 'L44', 'L45', 'L46', 'L47'
]
baby_cols = [
    'u55', 'u54', 'u53', 'u52', 'u51', 'u61', 'u62', 'u63', 'u64', 'u65', 
    'l75', 'l74', 'l73', 'l72', 'l71', 'l81', 'l82', 'l83', 'l84', 'l85'
]

def calculate_comprehensive_metrics(row):
    p_teeth = row[perm_cols]
    b_teeth = row[baby_cols]
    all_teeth = pd.concat([p_teeth, b_teeth])
    
    Perm_D = (p_teeth == 3).sum() + (p_teeth == 8).sum()
    Perm_M = (p_teeth == 4).sum()
    Perm_F = (p_teeth == 1).sum()
    Perm_DMFT = Perm_D + Perm_M + Perm_F
    
    Baby_d = (b_teeth == 3).sum() + (b_teeth == 8).sum()
    Baby_m = (b_teeth == 4).sum()
    Baby_f = (b_teeth == 1).sum()
    Baby_DMFT = Baby_d + Baby_m + Baby_f

    dmft_total_score = (Perm_D + Perm_M + Perm_F) + (Baby_d + Baby_m + Baby_f)
    
    return pd.Series({
        'Perm_D': Perm_D, 'Perm_M': Perm_M, 'Perm_F': Perm_F,
        'Baby_d': Baby_d, 'Baby_m': Baby_m, 'Baby_f': Baby_f,
        'Perm_DMFT': Perm_DMFT,
        'Baby_DMFT': Baby_DMFT,
        'DMFT_Index': dmft_total_score
    })

# Apply function
print("Calculating metrics...")
metrics_df = data0.apply(calculate_comprehensive_metrics, axis=1)
df_final = pd.concat([data0, metrics_df], axis=1)

# --- Analysis ---

# 1. Calculate DFT_Index (Decayed + Filled, excluding Missing)
df_final['DFT_Index'] = df_final['Perm_D'] + df_final['Perm_F'] + df_final['Baby_d'] + df_final['Baby_f']

# 2. Create Period column
df_final['date'] = pd.to_datetime(df_final['date'])

def get_period(date):
    year = date.year
    if 2017 <= year <= 2019:
        return 'Pre-COVID'
    elif 2020 <= year <= 2022:
        return 'COVID'
    elif 2023 <= year <= 2025:
        return 'After COVID'
    else:
        return None

df_final['Period'] = df_final['date'].apply(get_period)

# Filter data
df_period = df_final.dropna(subset=['Period']).copy()
period_order = ['Pre-COVID', 'COVID', 'After COVID']

print(f"Data counts per period:\n{df_period['Period'].value_counts()}")

# 3. Plot Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_period, x='Period', y='DFT_Index', order=period_order)
plt.title('Distribution of DFT_Index by Period')
plt.savefig('dft_boxplot.png')
print("Boxplot saved to dft_boxplot.png")

# 4. Statistical Test
print("\n--- Kruskal-Wallis Test ---")
groups = [df_period[df_period['Period'] == p]['DFT_Index'] for p in period_order]
stat, p_value = stats.kruskal(*groups)
print(f"Statistic: {stat}, p-value: {p_value}")

if p_value < 0.05:
    print("\n--- Pairwise Dunn's Test (Bonferroni adjusted) ---")
    dunn = sp.posthoc_dunn(df_period, val_col='DFT_Index', group_col='Period', p_adjust='bonferroni')
    print(dunn)
else:
    print("\nNo significant difference found between groups.")
