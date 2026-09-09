import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 4))
plt.plot(k_range, cv_scores, "b-o")
plt.axvline(x=best_k, color="r", linestyle="--", label=f"Best k={best_k}")
plt.xlabel("k")
plt.ylabel("5-Fold CV Accuracy")
plt.title("Elbow Method for Optimal k")
plt.legend()
plt.grid(True)
plt.show()
