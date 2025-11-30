# step7_feature_importance.py
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

X_train = pd.read_csv("data/X_train_res.csv")  # columns used for training
rf = joblib.load("models/random_forest.joblib")

feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(20)
print("Top features:\n", feat_imp)
plt.figure(figsize=(8,6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
