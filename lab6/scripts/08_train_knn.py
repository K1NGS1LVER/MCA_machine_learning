from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_s, y_train)
y_pred_knn = knn.predict(X_test_s)
print(f"KNN trained with k={best_k}.")
