better = "Logistic Regression" if m_lr[3] >= m_knn[3] else f"KNN (k={best_k})"
print(f">>> Better classifier (by F1): {better}")
print(f">>> Best train-test split: 80/20")
print(f">>> Best LR threshold: {best_thresh}")
