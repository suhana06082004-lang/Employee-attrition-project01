# step5_train_models_improved.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# load training data (SMOTE-resampled)
X_train = pd.read_csv("data/X_train_res.csv")
y_train = pd.read_csv("data/y_train_res.csv").squeeze()

# Train improved logistic with scaler + more iterations
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(solver="saga", max_iter=3000, C=1.0, random_state=42))
])
pipe.fit(X_train, y_train)
joblib.dump(pipe, "models/logistic_model_improved.joblib")

# Train random forest (unchanged)
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "models/random_forest.joblib")

print("Improved logistic + RF trained and saved.")
