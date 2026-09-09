from sklearn.metrics import confusion_matrix

print("Confusion Matrix - Logistic Regression:")
print(confusion_matrix(y_test, y_pred_lr))
print(f"\nConfusion Matrix - KNN (k={best_k}):")
print(confusion_matrix(y_test, y_pred_knn))
