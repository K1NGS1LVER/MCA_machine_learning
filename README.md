<div align="center">

# MCA521-4: Machine Learning Laboratory
### Master of Computer Applications (IV Trimester, 2026)
**Department of Computer Science | CHRIST (Deemed to be University), Bangalore**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

</div>

---

## 👨‍🎓 Candidate & Evaluation Profile

| Field | Details |
|:---|:---|
| **Student Name** | **Daniel Paul** |
| **Register Number** | **2547159** |
| **Class & Section** | **4MCA 'A'** |
| **Programme** | Master of Computer Applications (MCA) |
| **Course Code & Title** | **MCA521-4: Machine Learning** (Theo & Prac, 4 Credits) |
| **Faculty Evaluators** | **Dr. Rupali Sunil Wagh**, **Dr. Helen K Joy**, **Dr. Shivangi Singh** |
| **Academic Year** | 2026 |

---

## 🎯 Course Outcomes & Bloom's Taxonomy Mapping

This laboratory repository covers practical implementations strictly mapped against the curriculum's Course Outcomes (**CO1–CO5**) and Revised Bloom's Taxonomy (**RBT**) levels:

| Lab # | Experiment Title | RBT Level | Course Outcome | Primary Dataset | Evaluation Focus |
|:---:|:---|:---:|:---:|:---|:---|
| **01** | **Data Preprocessing & Exploratory Visualization** | L3 (Apply) | **CO1, CO3** | `city_day.csv`, `crop_production.csv` | Missing value imputation, IQR Winsorization, distribution skewness |
| **02** | **Time-Series Exploration & Cross-Domain Inferences** | L3 (Apply) | **CO2, CO3** | Cleaned AQI & Agricultural Yields | Yearly trends (2015–2020), multi-city pollutant ANOVA, Pearson $r$ correlation |
| **03** | **Simple & Multiple Linear Regression with OLS & Parameter Persistence** | L3 (Apply) | **CO3** | `student_survey.csv` | Normal Equation closed-form vs Gradient Solvers, residual normality, Pickle serialization |
| **04** | **KNN Classification & Distance Metric Comparison vs Regression Metrics** | L3 (Apply) | **CO3** | Breast Cancer Wisconsin (Diagnostic) | Distance metrics ($L_1, L_2, L_\infty$), 10-Fold Stratified CV, ROC/PR trade-offs |
| **05** | **Batch Gradient Descent Optimization from Mathematical First Principles** | L3 (Apply) | **CO3** | UCI Student Performance (`student-mat.csv`) | Cost function $J(\theta)$ minimization, learning rate $\alpha$ sweeps, feature standardization |
| **06** | **Empirical Classifier Comparison: Logistic Regression vs KNN** | L3, L4 | **CO3, CO4** | Breast Cancer Wisconsin | Class imbalance handling, confusion matrix cost analysis, decision threshold sweeping |
| **07** | **Decision Tree Induction, Information Theory & Cost-Complexity Pruning** | L4 (Analyze) | **CO3** | Fisher's Iris Multi-Class | Shannon Entropy / Information Gain vs Gini Impurity, tree pruning, decision surface |
| **08** | **Categorical Naive Bayes Classification & Laplace Smoothing** | L3 (Apply) | **CO3** | Weather / Play Tennis Dataset | Prior & conditional probabilities, Laplace smoothing ($\alpha$), feature independence |
| **09** | **Support Vector Machines (SVM) & Dimensionality Reduction (PCA / LDA)** | L3, L4 | **CO4** | Breast Cancer & Wine Varietals | Maximal margin hyperplanes, RBF kernel trick, PCA Scree plot, LDA discriminant axes |
| **10** | **Non-Linear XOR Separation via Multi-Layer Perceptrons (MLP)** | L5 (Evaluate) | **CO4, CO5** | Augmented Synthetic XOR (400 samples) | Non-linear decision surfaces, multi-framework benchmark (**Keras**, **PyTorch**, **TF**) |

---

## 🔬 Lab-by-Lab Mathematical Foundations & Experimental Deep Dives

<details>
<summary><b>📂 Lab 1: Data Preprocessing & Exploratory Visualization</b> <i>(Click to expand)</i></summary>

- **Objective:** Ingest raw, heterogeneous environmental datasets (`city_day.csv` with 29,531 records and `crop_production.csv` with 246,091 records), assess data hygiene, treat missingness, and handle extreme outliers.
- **Mathematical Formulations:**
  - **IQR Outlier Boundaries:** $\text{IQR} = Q_3 - Q_1$, $\text{Lower} = Q_1 - 1.5\times\text{IQR}$, $\text{Upper} = Q_3 + 1.5\times\text{IQR}$.
  - **Winsorization (Capping):** $x_i' = \min(\max(x_i, \text{Lower}), \text{Upper})$.
- **Key Findings:**
  - Preserved critical winter pollution spikes by applying **99th-percentile / IQR Winsorization** rather than row deletion, preventing systemic temporal bias in downstream time-series modeling.
- **Source Code:** [`lab1/process.ipynb`](./lab1/process.ipynb)
</details>

<details>
<summary><b>📂 Lab 2: Time-Series Exploration & Cross-Domain Inferences</b> <i>(Click to expand)</i></summary>

- **Objective:** Answer longitudinal policy questions (evaluating post-2018 National Clean Air Programme effects) and determine whether deteriorating AQI correlates with declining agricultural crop yields.
- **Statistical Measures:**
  - **Pearson Correlation Coefficient:** $r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$
- **Key Findings:**
  - Longitudinal trend revealed a measurable mean AQI decline post-2018 ($\approx -14.2\%$), with pronounced seasonality driven by winter thermal inversions in northern Indo-Gangetic cities (Delhi, Lucknow).
- **Source Code:** [`lab1/process.ipynb` (Tasks 6–10)](./lab1/process.ipynb)
</details>

<details>
<summary><b>📂 Lab 3: Simple & Multiple Linear Regression with OLS & Parameter Persistence</b> <i>(Click to expand)</i></summary>

- **Objective:** Predict academic performance (GPA) from continuous behavioral indicators (CIA %, Attendance %) using closed-form Ordinary Least Squares and Scikit-Learn.
- **Mathematical Formulation:**
  - **Closed-Form OLS Normal Equation:** $\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
  - **Univariate Slope & Intercept:** $\hat{\beta}_1 = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}, \quad \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$
- **Validation & Model Persistence:**
  - Scikit-Learn and manual OLS yielded mathematically identical weights ($|\Delta| < 10^{-12}$).
  - Saved model weights and scaler pipelines via `pickle` / `joblib` for zero-overhead inference.
- **Source Code:** [`lab3/lab3.ipynb`](./lab3/lab3.ipynb) | [`lab3/lab3_extension.ipynb`](./lab3/lab3_extension.ipynb)
</details>

<details>
<summary><b>📂 Lab 4: KNN Classification & Distance Metric Comparison vs Regression Metrics</b> <i>(Click to expand)</i></summary>

- **Objective:** Evaluate $K$-Nearest Neighbours on high-dimensional clinical diagnostic data (Breast Cancer Wisconsin, 30 features, 569 samples) across multiple vector distance topologies.
- **Distance Topologies Evaluated:**
  - **Euclidean ($L_2$):** $d(\mathbf{u}, \mathbf{v}) = \sqrt{\sum (u_i - v_i)^2}$
  - **Manhattan ($L_1$):** $d(\mathbf{u}, \mathbf{v}) = \sum |u_i - v_i|$
  - **Chebyshev ($L_\infty$):** $d(\mathbf{u}, \mathbf{v}) = \max_i |u_i - v_i|$
- **Quantitative Results:**
  - **Optimal Hyperparameter:** $K = 5$ (selected via 10-fold Stratified Cross-Validation).
  - **Mean Test Accuracy:** $97.37\%$ | **ROC-AUC:** $0.994$ | **$F_1$ Score:** $0.979$.
- **Source Code & Plots:** [`lab4/lab4.ipynb`](./lab4/lab4.ipynb) | [`lab4/roc_curve.png`](./lab4/roc_curve.png)
</details>

<details>
<summary><b>📂 Lab 5: Batch Gradient Descent from Mathematical First Principles</b> <i>(Click to expand)</i></summary>

- **Objective:** Implement Batch Gradient Descent from scratch to minimize Mean Squared Error cost $J(\boldsymbol{\theta})$ on the UCI Student Performance dataset.
- **Optimization Formulation:**
  - **Cost Function:** $J(\boldsymbol{\theta}) = \frac{1}{2m} \sum_{i=1}^m \left( h_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) - y^{(i)} \right)^2$
  - **Parameter Update Rule:** $\boldsymbol{\theta} := \boldsymbol{\theta} - \alpha \frac{1}{m} \mathbf{X}^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$
- **Convergence Dynamics:**
  - Evaluated learning rates $\alpha \in \{0.0001, 0.001, 0.01, 0.1\}$.
  - Standardized features ($\mu=0, \sigma=1$) were mathematically proven necessary to prevent gradient oscillation and catastrophic numerical overflow.
- **Source Code:** [`lab5/lab5_executed.ipynb`](./lab5/lab5_executed.ipynb)
</details>

<details>
<summary><b>📂 Lab 6: Empirical Classifier Comparison: Logistic Regression vs KNN</b> <i>(Click to expand)</i></summary>

- **Objective:** Compare parametric Logistic Regression against non-parametric KNN under varying decision thresholds and train-test splits.
- **Mathematical Formulations:**
  - **Sigmoid Activation:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - **Binary Cross-Entropy Loss:** $\mathcal{L}(\theta) = -\frac{1}{m}\sum \left[ y^{(i)}\log(\hat{y}^{(i)}) + (1 - y^{(i)})\log(1 - \hat{y}^{(i)}) \right]$
- **Key Findings:**
  - Swept decision thresholds $\tau \in [0.1, 0.9]$. In medical diagnosis where False Negatives (FN) carry high penalty, setting $\tau = 0.35$ maximized diagnostic Recall ($0.985$) with minimal Precision sacrifice.
- **Source Code:** [`lab6/lab6.ipynb`](./lab6/lab6.ipynb)
</details>

<details>
<summary><b>📂 Lab 7: Decision Tree Induction, Information Theory & Pruning</b> <i>(Click to expand)</i></summary>

- **Objective:** Construct multi-class Decision Trees on the Iris dataset, comparing splitting criteria and cost-complexity pruning.
- **Information Theory Formulations:**
  - **Shannon Entropy:** $H(S) = -\sum_{i=1}^C p_i \log_2(p_i)$
  - **Information Gain:** $IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$
  - **Gini Impurity:** $I_G(p) = 1 - \sum_{i=1}^C p_i^2$
- **Findings:**
  - Gini Impurity converged faster computationally while achieving identical tree topology to Shannon Entropy on continuous split boundaries. Pre-pruning at $\text{max\_depth}=3$ prevented leaf overfitting.
- **Source Code:** [`lab7/lab7_iris_decision_tree.ipynb`](./lab7/lab7_iris_decision_tree.ipynb)
</details>

<details>
<summary><b>📂 Lab 8: Categorical Naive Bayes Classification & Laplace Smoothing</b> <i>(Click to expand)</i></summary>

- **Objective:** Implement Categorical Naive Bayes classifier on discrete categorical environmental data and analyze zero-frequency probability mitigation.
- **Bayesian Formulation:**
  - **Posterior Probability:** $P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{i=1}^n P(x_i \mid C_k)$
  - **Laplace Smoothing ($\alpha$):** $P(x_i = v \mid C_k) = \frac{N_{k, v} + \alpha}{N_k + \alpha \cdot K_i}$
- **Key Findings:**
  - Validated that without Laplace correction ($\alpha=0$), unseen feature combinations cause zero-probability collapse ($P(\mathbf{x}|C_k)=0$). Setting $\alpha=1.0$ guarantees robust generalized inference.
- **Source Code:** [`lab8/lab8.ipynb`](./lab8/lab8.ipynb)
</details>

<details>
<summary><b>📂 Lab 9: Support Vector Machines (SVM) & Dimensionality Reduction (PCA / LDA)</b> <i>(Click to expand)</i></summary>

- **Objective:** Contrast maximal margin supervised hyperplanes against unsupervised orthogonal projection (PCA) and supervised linear discriminant analysis (LDA).
- **Mathematical Formulations:**
  - **SVM Soft-Margin Dual:** $\max_\alpha \sum \alpha_i - \frac{1}{2}\sum \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \ 0 \le \alpha_i \le C$
  - **RBF Kernel:** $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$
  - **PCA Eigendecomposition:** $\mathbf{\Sigma} = \frac{1}{n}\mathbf{X}^T \mathbf{X} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^T$
- **Quantitative Findings:**
  - 2 Principal Components explained $78.4\%$ of total variance across 30 features in Breast Cancer data.
  - LDA maximized between-class scatter matrix $S_B$ relative to within-class scatter $S_W$, producing superior class separation in 2D latent space.
- **Source Code:** [`lab9/lab9_notebook.ipynb`](./lab9/lab9_notebook.ipynb) | [`lab9/lab9_report.pdf`](./lab9/lab9_report.pdf)
</details>

<details>
<summary><b>📂 Lab 10: Non-Linear XOR Separation via Multi-Layer Perceptrons (MLP)</b> <i>(Click to expand)</i></summary>

- **Objective:** Solve the classic non-linearly separable Minsky-Papert XOR problem using a 2-layer Multi-Layer Perceptron implemented across **Keras**, **PyTorch**, and **TensorFlow low-level API**.
- **Architecture & Forward Equations:**
  - **Hidden Layer (Tanh):** $\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}, \quad \mathbf{a}^{(1)} = \tanh(\mathbf{z}^{(1)})$
  - **Output Layer (Sigmoid):** $z^{(2)} = \mathbf{W}^{(2)}\mathbf{a}^{(1)} + b^{(2)}, \quad \hat{y} = \sigma(z^{(2)})$
- **Multi-Framework Cross-Validation Benchmark:**
  - **Sample Size:** 400 augmented samples ($\sigma=0.1$ Gaussian noise, perfectly balanced 200/200).
  - **Validation:** 5-Fold Stratified Cross-Validation.
  - **Keras Test Accuracy:** $100.0\%$ | **PyTorch Test Accuracy:** $100.0\%$ | **TF Low-Level Test Accuracy:** $100.0\%$.
- **Source Code & Visuals:** [`lab10/make_report.py`](./lab10/make_report.py) | [`lab10/figs/decision_boundaries.png`](./lab10/figs/decision_boundaries.png)
</details>

---

## 🌟 Continuous Internal Assessment (CIA) Projects

### 🚗 [CIA-1: Road Accident Severity & Economic Impact Analysis](./cia_1/)
- **Core Focus:** Comprehensive exploratory modeling of road traffic accidents, analyzing fatal injury distributions, non-linear traffic volume correlations, and economic loss projections.
- **Key Artifacts:** Heatmaps ([`correlation_heatmap.png`](./cia_1/correlation_heatmap.png)), boxplots ([`financial_impact_boxplots.png`](./cia_1/financial_impact_boxplots.png)), notebook ([`tes.ipynb`](./cia_1/tes.ipynb)).

### 🚨 [CIA-3: Mission Crisis — Disaster Response Message Classification Pipeline](./cia3/)
- **Problem Statement:** Real-time triage of 30,000+ multi-lingual emergency disaster messages into actionable humanitarian relief categories (medical aid, search & rescue, water supply).
- **Machine Learning Architecture:**
  - **Text Vectorization:** Sublinear TF-IDF N-gram feature representation ($1 \le n \le 2$).
  - **Classifiers:** Multi-Output Logistic Regression, Balanced Random Forests, and XGBoost.
  - **Evaluation Metric:** Macro $F_1$-score with Stratified 5-Fold Cross-Validation.
---

## 🎓 End-Term Practical Examination Solutions

### 🏡 [Question 4(i): California Housing Price Prediction (PCA + SVR)](./end_term/)
- **Core Focus:** End-to-end regression pipeline combining target distribution analysis, IQR outlier inspection, correlation heatmap, PCA dimensionality reduction via scree plot (5 components >95% variance), and Support Vector Regression with RBF kernel.
- **Key Enhancements:** 5-Fold Cross-Validation, `StandardScaler` vs `RobustScaler` sensitivity analysis, `GridSearchCV` hyperparameter tuning ($C, \epsilon$), and 3-panel residual diagnostics (Actual vs Predicted, Residuals vs Fitted, Normal Q-Q plot).
- **Artifacts:** Executed notebook ([`California_Housing_PCA_SVR_Pipeline.ipynb`](./end_term/California_Housing_PCA_SVR_Pipeline.ipynb)) | Rendered report ([`California_Housing_PCA_SVR_Pipeline.pdf`](./end_term/California_Housing_PCA_SVR_Pipeline.pdf)).

### 🌸 [Question 5: Multi-Layer Perceptron (MLP) Classifier on Iris Dataset](./end_term/)
- **Core Focus:** Multi-class classification comparing single-layer `(100,)` vs two-layer `(50, 30)` MLP architectures with Stratified 5-Fold Cross-Validation wrapped in leakage-free pipelines.
- **Key Enhancements:** Feature-level KDE class separability analysis, training loss convergence curves, train-vs-test generalization gap quantification, baseline model benchmarking (vs Logistic Regression & Random Forest), `GridSearchCV`, permutation feature importance, and normalized recall confusion matrices.
- **Artifacts:** Executed notebook ([`Iris_MLP_Architecture_Comparison.ipynb`](./end_term/Iris_MLP_Architecture_Comparison.ipynb)) | Rendered report ([`Iris_MLP_Architecture_Comparison.pdf`](./end_term/Iris_MLP_Architecture_Comparison.pdf)).

---

## 🚀 Environment Setup & Reproducibility

### 1. Repository Clone
```bash
git clone https://github.com/K1NGS1LVER/MCA_machine_learning.git
cd MCA_machine_learning
```

### 2. Environment Initialization
```bash
# Using Python 3.10+
python3 -m venv venv
source venv/bin/activate

# Install all locked dependencies
pip install -r requirements.txt
```

### 3. Lab Manual Compilation
The laboratory manual (`.docx` and `.pdf`) is compiled automatically directly from executed notebooks:

```bash
# Generate the official Word document
python generate_perfect_manual.py

# Render high-resolution PDF with Pandoc and Weasyprint
pandoc Daniel_Paul_2547159_ML_Lab_Manual.docx -o Daniel_Paul_2547159_ML_Lab_Manual.pdf --pdf-engine=weasyprint
```

---

## 📑 Repository Structure

```
.
├── README.md                                # Comprehensive academic documentation & portfolio index
├── .gitignore                               # Environment, OS metadata, and checkpoint ignore rules
├── requirements.txt                         # Pinned Python package dependencies
├── Daniel_Paul_2547159_ML_Lab_Manual.docx   # Master compiled Word Lab Manual
├── Daniel_Paul_2547159_ML_Lab_Manual.pdf    # Master compiled PDF Lab Manual (183 pages)
├── generate_perfect_manual.py               # Programmatic docx compiler script
│
├── lab1/                                    # Lab 1 & 2: Data Preprocessing, IQR Winsorization & AQI Trends
│   ├── process.ipynb                        # Tasks 1 to 10 implementation
│   ├── city_day.csv                         # CPCB National AQI Dataset
│   └── crop_production.csv                  # Ministry of Agriculture Dataset
│
├── lab3/                                    # Lab 3: Linear Regression & Parameter Persistence
│   ├── lab3.ipynb                           # Simple Linear Regression (OLS vs Sklearn)
│   ├── lab3_extension.ipynb                 # Multiple Linear Regression & Overfitting study
│   ├── student_survey.csv                   # Cleaned student dataset
│   └── linear_regression_weights.pkl        # Serialized model parameters
│
├── lab4/                                    # Lab 4: KNN Classification & Evaluation Metrics
│   ├── lab4.ipynb                           # Distance metrics, K-Elbow, ROC/PR curves
│   └── *.png                                # Confusion matrix, accuracy vs K, ROC curves
│
├── lab5/                                    # Lab 5: Batch Gradient Descent from Scratch
│   ├── lab5_executed.ipynb                  # GD optimization, learning rate tuning
│   └── student-mat.csv                      # UCI Student Performance Dataset
│
├── lab6/                                    # Lab 6: Logistic Regression vs KNN Classifiers
│   └── lab6.ipynb                           # Decision threshold tuning, class imbalance study
│
├── lab7/                                    # Lab 7: Decision Tree Classification & Pruning
│   └── lab7_iris_decision_tree.ipynb        # Entropy vs Gini, tree depth pruning on Iris
│
├── lab8/                                    # Lab 8: Categorical Naive Bayes
│   ├── lab8.ipynb                           # Prior/posterior calculations, Laplace smoothing
│   └── weather.csv                          # Play Tennis benchmark dataset
│
├── lab9/                                    # Lab 9: SVM, PCA & LDA Dimensionality Reduction
│   ├── lab9_notebook.ipynb                  # SVM margin hyperplanes, Scree plots, LDA
│   └── figs/                                # High-resolution PCA/LDA visualization figures
│
├── lab10/                                   # Lab 10: Multi-Layer Perceptron XOR Non-Linear Separation
│   ├── make_report.py                       # Keras, PyTorch, TF low-level benchmark
│   └── figs/                                # Class distribution & non-linear decision contours
│
├── cia_1/                                   # CIA-1: Road Accident Analysis & Economic Loss
│   ├── tes.ipynb                            # Accident distribution analysis
│   └── *.png                                # Economic loss scatter plots & correlation grids
│
├── end_term/                                # End-Term Practical Examination Solutions
│   ├── California_Housing_PCA_SVR_Pipeline.ipynb # Q4(i): PCA + SVR with Scree, GridSearch & Residual Diagnostics
│   ├── California_Housing_PCA_SVR_Pipeline.pdf   # Q4(i): Rendered PDF report with all diagnostic plots
│   ├── Iris_MLP_Architecture_Comparison.ipynb    # Q5: MLP (100,) vs (50,30) with Loss Curves & Baseline Benchmarks
│   └── Iris_MLP_Architecture_Comparison.pdf      # Q5: Rendered PDF report with all confusion & KDE plots
│
└── cia3/                                    # CIA-3: Disaster Message Triage NLP Pipeline
    ├── mission_crisis_pipeline.ipynb        # TF-IDF + Ensemble classifier pipeline
    ├── mission_crisis_report.pdf            # Formal technical project report
    └── slides.pdf                           # Pitch deck presentation assets
```

---

## ⚖️ Academic Integrity & Department Standards
This repository adheres to the academic guidelines and ethical computing regulations established by the **Department of Computer Science, CHRIST (Deemed to be University)**. All models, mathematical formulations, and diagnostic evaluations have been developed, analyzed, and validated independently.
