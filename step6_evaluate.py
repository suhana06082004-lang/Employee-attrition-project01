# step6_evaluate.py
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# load test set
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# load models
models = {
    "Logistic_improved": joblib.load("models/logistic_model_improved.joblib"),
    "RandomForest": joblib.load("models/random_forest.joblib")
}

plt.figure(figsize=(7,6))
for name, m in models.items():
    # if pipeline that includes scaler, it accepts raw X_test
    y_pred = m.predict(X_test)
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)
    # ROC AUC if predict_proba available
    if hasattr(m, "predict_proba"):
        y_prob = m.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
        print("ROC AUC:", round(auc, 4))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

plt.plot([0,1],[0,1],"--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()
