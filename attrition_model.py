# step1_load_inspect.py
import pandas as pd

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print("Rows,Cols:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values per column:")
print(df.isna().sum())
print("\nAttrition counts:")
print(df['Attrition'].value_counts())
