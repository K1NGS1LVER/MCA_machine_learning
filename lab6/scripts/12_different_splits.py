from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

splits = [0.2, 0.25, 0.3, 0.35]
split_results = []

for test_size in splits:
    Xtr, Xte, ytr, yte = train_test_split(X_res, y_res, test_size=test_size, random_state=42, stratify=y_res)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xte_s = sc.transform(Xte)

    lr_temp = LogisticRegression(max_iter=10000, random_state=42).fit(Xtr_s, ytr)
    yhat_lr = lr_temp.predict(Xte_s)

    knn_temp = KNeighborsClassifier(n_neighbors=best_k).fit(Xtr_s, ytr)
    yhat_knn = knn_temp.predict(Xte_s)

    split_results.append({
        "Split": f"{int((1-test_size)*100)}/{int(test_size*100)}",
        "LR Acc": accuracy_score(yte, yhat_lr),
        "LR F1": f1_score(yte, yhat_lr),
        "KNN Acc": accuracy_score(yte, yhat_knn),
        "KNN F1": f1_score(yte, yhat_knn),
    })

fmt2 = "{:<10} {:>10} {:>10} {:>10} {:>10}"
print(fmt2.format("Split", "LR Acc", "LR F1", "KNN Acc", "KNN F1"))
print("-" * 52)
for r in split_results:
    print(fmt2.format(r["Split"], f"{r['LR Acc']:.4f}", f"{r['LR F1']:.4f}", f"{r['KNN Acc']:.4f}", f"{r['KNN F1']:.4f}"))
