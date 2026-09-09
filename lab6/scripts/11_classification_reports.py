from sklearn.metrics import classification_report

print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr, target_names=data.target_names))
print(f"KNN (k={best_k}):")
print(classification_report(y_test, y_pred_knn, target_names=data.target_names))
