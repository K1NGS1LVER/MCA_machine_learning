import numpy as np
from imblearn.over_sampling import RandomOverSampler

counts = np.bincount(y)
print(f"Before: benign={counts[1]}, malignant={counts[0]}")

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

counts = np.bincount(y_res)
print(f"After:  benign={counts[1]}, malignant={counts[0]}")
