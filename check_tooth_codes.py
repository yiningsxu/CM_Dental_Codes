import pandas as pd
df = pd.read_csv('/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis/data/analysisData_20260211.csv')
print("Unique values in U11:", df['U11'].unique())
print("Unique values in U16:", df['U16'].unique())
print("Unique values in u51:", df['u51'].unique())
print("Unique values in u55:", df['u55'].unique())
