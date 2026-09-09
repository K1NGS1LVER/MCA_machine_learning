import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot([r["Threshold"] for r in thresh_results], [r["F1"] for r in thresh_results], "b-o")
plt.axvline(x=best_thresh, color="r", linestyle="--", label=f"Best threshold={best_thresh}")
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("LR Threshold vs F1 Score")
plt.legend()
plt.grid(True)
plt.show()
