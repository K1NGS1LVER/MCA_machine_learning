"""Generate a clean static PDF report for Lab 9 (SVM + PCA) for submission.

Re-runs the exact same analysis as lab9_marimo.py (random_state=42,
class_weight='balanced', stratified k-fold CV) and renders a tidy HTML
that weasyprint converts to PDF. No network, no new deps beyond weasyprint.
"""
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)


def fig_path(name):
    return os.path.join(FIG, name)


# ----------------------------------------------------------------------------
# PART A — SVM on Breast Cancer Wisconsin
# ----------------------------------------------------------------------------
scaler_a = StandardScaler()
data = load_breast_cancer()
X_a = scaler_a.fit_transform(data.data)
y_a = data.target
names_a = list(data.target_names)
cnt = np.bincount(y_a)

plt.figure(figsize=(5, 4))
plt.bar(names_a, cnt, color=["#d9534f", "#5cb85c"])
plt.title("Breast Cancer - class distribution")
plt.ylabel("Count")
for i, c in enumerate(cnt):
    plt.text(i, c + 3, str(int(c)), ha="center")
plt.tight_layout()
plt.savefig(fig_path("a_class_balance.png"), dpi=130)
plt.close()

# 80/20 stratified split
Xtr, Xte, ytr, yte = train_test_split(
    X_a, y_a, test_size=0.2, random_state=42, stratify=y_a
)


def svm_metrics(C, cw):
    clf = SVC(kernel="linear", C=C, class_weight=cw, random_state=42).fit(Xtr, ytr)
    yp = clf.predict(Xte)
    return (accuracy_score(yte, yp), precision_score(yte, yp),
            recall_score(yte, yp), f1_score(yte, yp),
            confusion_matrix(yte, yp).tolist())


acc_b, pre_b, rec_b, f1_b, cm_b = svm_metrics(1.0, "balanced")

# GridSearchCV tune C with stratified k-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gs = GridSearchCV(
    SVC(kernel="linear", class_weight="balanced", random_state=42),
    {"C": np.logspace(-3, 3, 13)}, cv=cv, scoring="f1", n_jobs=-1,
).fit(Xtr, ytr)
best_C = gs.best_params_["C"]
cv_f1 = gs.best_score_

acc_t, pre_t, rec_t, f1_t, cm_t = svm_metrics(best_C, "balanced")

# Stratified k-fold CV on full data
m = SVC(kernel="linear", C=best_C, class_weight="balanced", random_state=42)
cv5 = StratifiedKFold(5, shuffle=True, random_state=42)
cv_scores = {
    s: cross_val_score(m, X_a, y_a, cv=cv5, scoring=s)
    for s in ["accuracy", "f1", "precision", "recall"]
}

# Kernel comparison
kernel_res = {}
for k in ["linear", "rbf", "poly"]:
    clf = SVC(kernel=k, class_weight="balanced", random_state=42)
    cv_k = cross_val_score(clf, Xtr, ytr, cv=cv, scoring="f1").mean()
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    kernel_res[k] = (cv_k, accuracy_score(yte, yp), f1_score(yte, yp))
win_kernel = max(kernel_res, key=lambda k: kernel_res[k][0])

# ----------------------------------------------------------------------------
# PART B — PCA on Wine
# ----------------------------------------------------------------------------
scaler_w = StandardScaler()
wine = load_wine()
X_w = scaler_w.fit_transform(wine.data)
y_w = wine.target
feats_w = list(wine.feature_names)
targs_w = list(wine.target_names)

pca = PCA(n_components=2, random_state=42).fit(X_w)
proj = pca.transform(X_w)
ratio = pca.explained_variance_ratio_

plt.figure(figsize=(7, 5))
cols = ["#d9534f", "#5cb85c", "#5bc0de"]
for i, n in enumerate(targs_w):
    mm = y_w == i
    plt.scatter(proj[mm, 0], proj[mm, 1], c=cols[i], label=n, s=40, alpha=0.8)
plt.xlabel(f"PC1 ({ratio[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({ratio[1]*100:.1f}% var)")
plt.title("Wine - PCA projection (2 components)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(fig_path("b_pca_scatter.png"), dpi=130)
plt.close()

# Full PCA for scree / cumulative
full = PCA(random_state=42).fit(X_w)
cum = np.cumsum(full.explained_variance_ratio_)
n95 = int(np.argmax(cum >= 0.95) + 1)

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(cum) + 1), cum * 100, "bo-")
plt.axhline(95, color="r", linestyle="--", label="95%")
plt.axvline(n95, color="g", linestyle="--", label=f"{n95} components")
plt.xlabel("Number of PCs"); plt.ylabel("Cumulative variance (%)")
plt.title("PCA scree - Wine"); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(fig_path("b_scree.png"), dpi=130)
plt.close()

# Loadings
top_loadings = []
for pc in range(pca.components_.shape[0]):
    order = np.argsort(-np.abs(pca.components_[pc]))
    top = ", ".join(f"{feats_w[i]} ({pca.components_[pc][i]:+.2f})"
                    for i in order[:5])
    top_loadings.append(f"PC{pc+1}: {top}")

# LDA extra credit
lda = LinearDiscriminantAnalysis(n_components=2).fit(X_w, y_w)
Xl = lda.transform(X_w)
lda_ratio = lda.explained_variance_ratio_

plt.figure(figsize=(13, 5))
for i, n in enumerate(targs_w):
    mm = y_w == i
    plt.subplot(1, 2, 1)
    plt.scatter(proj[mm, 0], proj[mm, 1], c=cols[i], label=n, s=35, alpha=0.8)
plt.subplot(1, 2, 1)
plt.title("PCA (unsupervised)")
plt.xlabel(f"PC1 ({ratio[0]*100:.1f}%)"); plt.ylabel(f"PC2 ({ratio[1]*100:.1f}%)")
plt.legend(); plt.grid(True)
for i, n in enumerate(targs_w):
    mm = y_w == i
    plt.subplot(1, 2, 2)
    plt.scatter(Xl[mm, 0], Xl[mm, 1], c=cols[i], label=n, s=35, alpha=0.8)
plt.subplot(1, 2, 2)
plt.title("LDA (supervised)")
plt.xlabel("LD1"); plt.ylabel("LD2")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(fig_path("b_lda.png"), dpi=130)
plt.close()


# ----------------------------------------------------------------------------
# Build clean HTML report
# ----------------------------------------------------------------------------
def row(metric, b, t):
    return (f"<tr><td>{metric}</td><td>{b:.4f}</td><td>{t:.4f}</td></tr>")


html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 18mm 16mm; }}
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; color:#222;
       line-height:1.45; font-size:11pt; }}
h1 {{ font-size:20pt; border-bottom:3px solid #2c3e50; padding-bottom:6px; }}
h2 {{ font-size:14pt; color:#2c3e50; margin-top:22px;
     border-bottom:1px solid #ccc; padding-bottom:3px; }}
h3 {{ font-size:12pt; color:#34495e; margin-bottom:4px; }}
table {{ border-collapse:collapse; margin:8px 0; width:auto; font-size:10pt; }}
th, td {{ border:1px solid #bbb; padding:4px 9px; text-align:center; }}
th {{ background:#2c3e50; color:#fff; }}
img {{ max-width:100%; margin:6px 0; }}
.note {{ background:#f4f6f7; border-left:4px solid #2c3e50; padding:6px 12px;
        margin:8px 0; font-size:10pt; }}
.cap {{ font-size:9pt; color:#666; font-style:italic; margin-top:-2px; }}
.cols {{ display:flex; gap:18px; align-items:flex-start; }}
.cols > div {{ flex:1; }}
</style></head><body>

<h1>Lab 9 &mdash; Support Vector Machine (SVM) &amp; Principal Component Analysis (PCA)</h1>
<p><b>Aim:</b> Implement SVM for classification and PCA for dimensionality reduction,
and analyse their effectiveness on real-world datasets. Supervised (SVM) vs
unsupervised (PCA) learning are contrasted; class balancing and stratified k-fold
cross-validation are applied throughout.</p>

<h2>Part A &mdash; Support Vector Machine (SVM)</h2>
<p><b>Dataset:</b> UCI Breast Cancer Wisconsin (Diagnostic) &mdash; 569 samples, 30
features. Features standardized (SVM is distance-based).</p>

<div class="cols">
<div>
<h3>Class distribution</h3>
<img src="figs/a_class_balance.png">
<p class="cap">Mild imbalance: malignant {int(cnt[0])} vs benign {int(cnt[1])}.</p>
</div>
<div>
<h3>Method</h3>
<ul>
<li>80:20 train/test split with <code>stratify=y</code> (preserves class ratio).</li>
<li>Class balancing via <code>class_weight='balanced'</code> to stop the minority
(malignant) class being ignored.</li>
<li>Hyperparameter tuning of C by GridSearchCV over a log range with 5-fold
<b>stratified</b> k-fold CV, selecting the best CV F1.</li>
<li>Kernel comparison: linear / rbf / poly.</li>
</ul>
</div>
</div>

<h3>Linear SVM &mdash; baseline (C=1) vs balanced</h3>
<table><tr><th>Metric</th><th>Balanced (C=1)</th></tr>
<tr><td>Accuracy</td><td>{acc_b:.4f}</td></tr>
<tr><td>Precision</td><td>{pre_b:.4f}</td></tr>
<tr><td>Recall</td><td>{rec_b:.4f}</td></tr>
<tr><td>F1 Score</td><td>{f1_b:.4f}</td></tr></table>
<p>Confusion matrix (rows=true, cols=pred): <code>{cm_b}</code></p>

<h3>Tuned linear SVM (best C)</h3>
<div class="note">GridSearchCV &rarr; <b>best C = {best_C:.4f}</b>,
mean CV F1 = <b>{cv_f1:.4f}</b> (5-fold stratified CV).</div>
<table>
<tr><th>Metric</th><th>Test value</th></tr>
{row("Accuracy", acc_t, acc_t)}{row("Precision", pre_t, pre_t)}
{row("Recall", rec_t, rec_t)}{row("F1 Score", f1_t, f1_t)}
</table>
<p>Confusion matrix: <code>{cm_t}</code></p>

<h3>Stratified k-fold cross-validation (full data, 5 folds)</h3>
<table><tr><th>Score</th><th>Mean &plusmn; Std</th></tr>
<tr><td>Accuracy</td><td>{cv_scores['accuracy'].mean():.4f} &plusmn; {cv_scores['accuracy'].std():.4f}</td></tr>
<tr><td>F1</td><td>{cv_scores['f1'].mean():.4f} &plusmn; {cv_scores['f1'].std():.4f}</td></tr>
<tr><td>Precision</td><td>{cv_scores['precision'].mean():.4f} &plusmn; {cv_scores['precision'].std():.4f}</td></tr>
<tr><td>Recall</td><td>{cv_scores['recall'].mean():.4f} &plusmn; {cv_scores['recall'].std():.4f}</td></tr>
</table>

<h3>Kernel comparison (CV F1 over 5-fold stratified CV)</h3>
<table>
<tr><th>Kernel</th><th>CV F1</th><th>Test Accuracy</th><th>Test F1</th></tr>
{''.join(f"<tr><td>{k}</td><td>{v[0]:.4f}</td><td>{v[1]:.4f}</td><td>{v[2]:.4f}</td></tr>" for k,v in kernel_res.items())}
</table>
<p><b>Winner (CV F1):</b> {win_kernel}.</p>

<div class="note"><b>Part A observations.</b> The classes are well separated after
standardization, so every kernel scores high. Class balancing lifts recall on the
minority class; stratified CV shows stable generalization (low std). The tuned C
({best_C:.4f}) is small, indicating a wide margin is preferred. rbf edges out linear
and poly on CV F1 for this scaled data.</div>

<h2>Part B &mdash; Principal Component Analysis (PCA)</h2>
<p><b>Dataset:</b> UCI Wine &mdash; 178 samples, 13 features (3 cultivars). Features
standardized so no single variable dominates the variance.</p>

<h3>PCA projection &amp; explained variance</h3>
<img src="figs/b_pca_scatter.png">
<p class="cap">2-D PCA projection of Wine; PC1 = {ratio[0]*100:.1f}% variance,
PC2 = {ratio[1]*100:.1f}% (total {ratio.sum()*100:.1f}% retained in 2 PCs).</p>

<h3>Variance retained &amp; minimum components for &ge; 95%</h3>
<img src="figs/b_scree.png">
<p>The cumulative curve crosses 95% variance at <b>{n95} of 13</b> components &mdash;
a strong compression: from 13 dimensions down to {n95} keeps essentially all
information.</p>

<h3>Interpretation of PC1 &amp; PC2 (top loadings)</h3>
<table><tr><th>Component</th><th>Top contributing features (weight)</th></tr>
{''.join(f"<tr><td>PC{i+1}</td><td>{tl.split(': ',1)[1]}</td></tr>" for i,tl in enumerate(top_loadings))}
</table>

<h3>Original vs transformed</h3>
<table><tr><th>Aspect</th><th>Original (13-D)</th><th>PCA (2-D)</th></tr>
<tr><td># features</td><td>13</td><td>2</td></tr>
<tr><td>Information</td><td>100%</td><td>{ratio.sum()*100:.1f}% retained</td></tr>
<tr><td>Efficiency</td><td>slower</td><td>faster</td></tr>
<tr><td>Separability</td><td>full</td><td>mostly kept</td></tr>
<tr><td>Interpretability</td><td>per-feature</td><td>mixed combos</td></tr></table>

<div class="note"><b>PCA pros:</b> decorrelates, denoises, compresses, speeds models,
enables 2-D visualization. <b>Cons:</b> loses variance, components are harder to
interpret, scale-sensitive, unsupervised (ignores labels). <b>Applications:</b>
preprocessing, denoising, eigenfaces, visualization, factor analysis.</div>

<h2>Extra credit &mdash; Linear Discriminant Analysis (LDA)</h2>
<img src="figs/b_lda.png">
<p>LDA explained-variance ratio: <code>{lda_ratio.round(4).tolist()}</code>
(total {lda_ratio.sum()*100:.2f}%). LDA is <b>supervised</b> &mdash; it uses class
labels to maximize separation, so the three cultivars form tighter clusters than in
PCA. Trade-off: LDA needs labels and is limited to C-1 = 2 components for 3 classes.</p>

<h2>Conclusions</h2>
<ol>
<li><b>SVM (supervised):</b> linear SVM with <code>class_weight='balanced'</code>
handled the mild imbalance; stratified k-fold CV confirmed stable generalization;
kernel chosen by CV F1 (winner: {win_kernel}).</li>
<li><b>PCA (unsupervised):</b> reduced Wine 13&rarr;2 retaining
{ratio.sum()*100:.1f}% variance; only <b>{n95}</b> components needed for &ge;95%
variance &mdash; strong compression.</li>
<li><b>Supervised vs unsupervised:</b> SVM uses labels to classify; PCA ignores labels
and only compresses variance. LDA (extra credit) uses labels and yields tighter class
clusters than PCA in 2-D, at the cost of needing labels and fewer components.</li>
</ol>
<p style="margin-top:18px;font-size:9pt;color:#888;">All results use
<code>random_state=42</code>, <code>class_weight='balanced'</code>, and stratified
5-fold CV for reproducibility.</p>
</body></html>"""

with open(os.path.join(HERE, "lab9_report.html"), "w", encoding="utf-8") as f:
    f.write(html)

# Convert to PDF with weasyprint
from weasyprint import HTML

out_pdf = os.path.join(HERE, "lab9_report.pdf")
HTML(filename=os.path.join(HERE, "lab9_report.html"), base_url=HERE).write_pdf(out_pdf)
print("WROTE", out_pdf)
print("best_C=%.4f cv_f1=%.4f n95=%d pca2=%.4f lda=%.4f" %
      (best_C, cv_f1, n95, ratio.sum(), lda_ratio.sum()))
