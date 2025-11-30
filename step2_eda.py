# step2_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
# target distribution
sns.countplot(x='Attrition', data=df)
plt.title('Attrition count (No vs Yes)')
plt.show()

# Attrition by OverTime
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title('Attrition by OverTime')
plt.show()

# MonthlyIncome distribution
sns.histplot(df['MonthlyIncome'], bins=40)
plt.title('MonthlyIncome distribution')
plt.show()

# correlation of numeric features
num_cols = df.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), cmap='coolwarm', center=0)
plt.title('Numeric features correlation')
plt.show()
