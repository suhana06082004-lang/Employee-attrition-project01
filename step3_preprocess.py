# step3_preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 1) Encode target (Yes->1, No->0)
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

# 2) Inspect categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns:", cat_cols)

# 3) One-hot encode categorical columns (safe for ML)
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 4) Save processed CSV for re-use
df.to_csv("data/processed_attrition.csv", index=False)
print("Processed shape:", df.shape)
