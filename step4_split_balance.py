# step4_split_balance.py
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv("data/processed_attrition.csv")
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split (stratify keeps class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)
print("Train class counts:\n", y_train.value_counts())

# Optional: apply SMOTE to training set (only training)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("After SMOTE train class counts:\n", y_train_res.value_counts())

# Save split files for next steps
X_train_res.to_csv("data/X_train_res.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train_res.to_csv("data/y_train_res.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
print("Saved split CSVs.")
