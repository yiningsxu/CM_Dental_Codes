import pandas as pd
import numpy as np
from scipy import stats

"""
Total children under 6 years old: 244

--- ECC (dmft >= 1) and D-ECC (dmft >= 4) Proportion by Abuse Category (Age < 6) ---
     Abuse Type   N  ECC (n) ECC (%)  D-ECC (n) D-ECC (%)
 Physical Abuse  79       17   21.5%          7      8.9%
        Neglect 111       40   36.0%         22     19.8%
Emotional Abuse  47       12   25.5%          7     14.9%
   Sexual Abuse   7        1   14.3%          1     14.3%

Overall (N=244):
ECC: 70 (28.7%)
D-ECC: 37 (15.2%)

--- Statistical Tests (Chi-square) ---
ECC across groups: p-value = 0.119
D-ECC across groups: p-value = 0.229
"""

def calculate_ecc_stats():
    # Load dataset
    file_path = "/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data/data_OnlyAbuse_N1235.csv"
    try:
        df = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # Filter for children under 6 years old
    # Using 'age_year' or 'age'. Let's check which one is better. 
    # Based on previous file view, 'age_year' seems to be the integer age.
    # Let's use 'age' (float) < 6.0 for precision if available, or 'age_year' < 6.
    # The snippet showed 'age' as float (e.g., 15.083333).
    
    df_u6 = df[df['age'] < 6].copy()
    
    print(f"Total children under 6 years old: {len(df_u6)}")
    
    if len(df_u6) == 0:
        print("No children under 6 found.")
        return

    # Define ECC and D-ECC
    # ECC: dmft >= 1 (Baby_DMFT)
    # D-ECC: dmft >= 4 (Baby_DMFT)
    
    # Ensure Baby_DMFT is numeric
    df_u6['Baby_DMFT'] = pd.to_numeric(df_u6['Baby_DMFT'], errors='coerce')
    
    df_u6['ECC'] = df_u6['Baby_DMFT'] >= 1
    df_u6['D_ECC'] = df_u6['Baby_DMFT'] >= 4
    
    # Group by abuse category
    abuse_groups = ["Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse"]
    
    print("\n--- ECC (dmft >= 1) and D-ECC (dmft >= 4) Proportion by Abuse Category (Age < 6) ---")
    
    results = []
    
    for abuse_type in abuse_groups:
        group_data = df_u6[df_u6['abuse'] == abuse_type]
        n = len(group_data)
        
        if n == 0:
            print(f"\n{abuse_type}: N=0")
            continue
            
        ecc_count = group_data['ECC'].sum()
        decc_count = group_data['D_ECC'].sum()
        
        ecc_prop = (ecc_count / n) * 100
        decc_prop = (decc_count / n) * 100
        
        results.append({
            'Abuse Type': abuse_type,
            'N': n,
            'ECC (n)': ecc_count,
            'ECC (%)': f"{ecc_prop:.1f}%",
            'D-ECC (n)': decc_count,
            'D-ECC (%)': f"{decc_prop:.1f}%"
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Overall
    n_total = len(df_u6)
    ecc_total = df_u6['ECC'].sum()
    decc_total = df_u6['D_ECC'].sum()
    print(f"\nOverall (N={n_total}):")
    print(f"ECC: {ecc_total} ({ecc_total/n_total*100:.1f}%)")
    print(f"D-ECC: {decc_total} ({decc_total/n_total*100:.1f}%)")

    # Chi-square test
    print("\n--- Statistical Tests (Chi-square) ---")
    
    # Create contingency tables
    # ECC
    contingency_ecc = pd.crosstab(df_u6['abuse'], df_u6['ECC'])
    chi2, p, dof, ex = stats.chi2_contingency(contingency_ecc)
    print(f"ECC across groups: p-value = {p:.3f}")
    
    # D-ECC
    contingency_decc = pd.crosstab(df_u6['abuse'], df_u6['D_ECC'])
    chi2, p, dof, ex = stats.chi2_contingency(contingency_decc)
    print(f"D-ECC across groups: p-value = {p:.3f}")

if __name__ == "__main__":
    calculate_ecc_stats()
