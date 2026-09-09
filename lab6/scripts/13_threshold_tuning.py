from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

lr_probs = lr.predict_proba(X_test_s)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
thresh_results = []

for t in thresholds:
    yhat_t = (lr_probs >= t).astype(int)
    thresh_results.append({
        "Threshold": round(t, 2),
        "Accuracy": accuracy_score(y_test, yhat_t),
        "Precision": precision_score(y_test, yhat_t),
        "Recall": recall_score(y_test, yhat_t),
        "F1": f1_score(y_test, yhat_t),
    })

best_thresh = max(thresh_results, key=lambda x: x["F1"])["Threshold"]

fmt3 = "{:<12} {:>10} {:>10} {:>10} {:>10}"
print(fmt3.format("Threshold", "Accuracy", "Precision", "Recall", "F1"))
print("-" * 52)
for r in thresh_results:
    mark = " <-- best" if r["Threshold"] == best_thresh else ""
    print(fmt3.format(r["Threshold"], f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}", f"{r['Recall']:.4f}", f"{r['F1']:.4f}") + mark)
