from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
print(f"569 samples, 30 features, 2 classes (benign/malignant)")
