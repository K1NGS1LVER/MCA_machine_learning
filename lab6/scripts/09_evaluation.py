from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calc(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
    )

m_lr = calc(y_test, y_pred_lr)
m_knn = calc(y_test, y_pred_knn)

fmt = "{:<22} {:>10} {:>10} {:>10} {:>10}"
print(fmt.format("Classifier", "Accuracy", "Precision", "Recall", "F1 Score"))
print("-" * 62)
print(fmt.format("Logistic Regression", *(f"{v:.4f}" for v in m_lr)))
print(fmt.format(f"KNN (k={best_k})", *(f"{v:.4f}" for v in m_knn)))
