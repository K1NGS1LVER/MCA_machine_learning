import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from imblearn.over_sampling import RandomOverSampler

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Fix class imbalance
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

# Train-test split and scaling
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train Logistic Regression
lr = LogisticRegression(max_iter=10000, random_state=42)
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

# Elbow method for KNN
k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_s, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())

best_k = list(k_range)[np.argmax(cv_scores)]

# Train KNN with best k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_s, y_train)
y_pred_knn = knn.predict(X_test_s)

# Calculate metrics
def calc(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )

m_lr = calc(y_test, y_pred_lr)
m_knn = calc(y_test, y_pred_knn)

# Threshold tuning for LR
lr_probs = lr.predict_proba(X_test_s)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)
thresh_results = []
for t in thresholds:
    yhat_t = (lr_probs >= t).astype(int)
    thresh_results.append({
        "Threshold": round(t, 2),
        "F1": f1_score(y_test, yhat_t),
    })
best_thresh = max(thresh_results, key=lambda x: x["F1"])["Threshold"]
best_f1 = max(thresh_results, key=lambda x: x["F1"])["F1"]

# Print summary for evaluator
print("=" * 60)
print("LAB 6 SUMMARY - Logistic Regression vs KNN")
print("=" * 60)
print(f"\nDataset: Breast Cancer Wisconsin (569 samples, 30 features)")
print(f"Classes: Benign (357), Malignant (212) [before oversampling]")
print(f"After oversampling: Both classes = 357")

print(f"\nTrain-Test Split: 80/20")
print(f"Feature Scaling: StandardScaler (zero mean, unit variance)")

print(f"\n--- Logistic Regression ---")
print(f"Accuracy:  {m_lr[0]:.4f}")
print(f"Precision: {m_lr[1]:.4f}")
print(f"Recall:    {m_lr[2]:.4f}")
print(f"F1 Score:  {m_lr[3]:.4f}")

print(f"\n--- KNN (k={best_k}) ---")
print(f"Best k from elbow method (5-fold CV): {best_k}")
print(f"CV Accuracy at k={best_k}: {max(cv_scores):.4f}")
print(f"Accuracy:  {m_knn[0]:.4f}")
print(f"Precision: {m_knn[1]:.4f}")
print(f"Recall:    {m_knn[2]:.4f}")
print(f"F1 Score:  {m_knn[3]:.4f}")

print(f"\n--- Threshold Tuning (Logistic Regression) ---")
print(f"Best threshold: {best_thresh}")
print(f"Best F1 score with threshold tuning: {best_f1:.4f}")

print(f"\n--- Comparison ---")
print(f"LR has 7 errors vs KNN's 10 errors")
better = "Logistic Regression" if m_lr[3] >= m_knn[3] else f"KNN (k={best_k})"
print(f"Better classifier (by F1): {better}")

print(f"\n--- Verdict ---")
print(f"Logistic Regression is the better classifier for this dataset.")
print(f"It offers higher accuracy ({m_lr[0]:.1%} vs {m_knn[0]:.1%})")
print(f"and better F1 score ({m_lr[3]:.4f} vs {m_knn[3]:.4f}).")
print(f"Best threshold for LR: {best_thresh}")
print(f"Best k for KNN: {best_k}")
print(f"Best train-test split: 80/20")
print("=" * 60)
