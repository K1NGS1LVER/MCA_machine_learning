# MCA521-4: Machine Learning Laboratory

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20PyTorch%20%7C%20TensorFlow-orange.svg)](https://scikit-learn.org/)
[![Course](https://img.shields.io/badge/Course-MCA521--4%20Machine%20Learning-green.svg)](#)
[![Institution](https://img.shields.io/badge/Institution-CHRIST%20(Deemed%20to%20be%20University)-red.svg)](https://christuniversity.in/)

Comprehensive laboratory coursework, practical implementations, and continuous assessments for **MCA521-4: Machine Learning** (IV Trimester MCA, 2026).

---

## 👨‍🎓 Student Details

- **Student Name:** Daniel Paul
- **Register Number:** 2547159
- **Class / Section:** 4MCA 'A'
- **Department:** Department of Computer Science, CHRIST (Deemed to be University), Bangalore
- **Faculty Evaluators:** Dr. Rupali Sunil Wagh, Dr. Helen K Joy, Dr. Shivangi Singh

---

## 📚 Lab Curriculum & Directory Map

| Lab # | Experiment Title | Source Directory | Primary Dataset | Key Techniques & Algorithms |
|:---:|:---|:---|:---|:---|
| **01** | **Data Preprocessing & Visualization** | [`lab1/`](./lab1/) | `city_day.csv`, `crop_production.csv` | Missing value imputation, duplicate removal, IQR Winsorization, pollutant distributions |
| **02** | **Data Exploration & Inferences** | [`lab1/`](./lab1/) | `city_day.csv`, `crop_production.csv` | Time-series trend analysis (2015–2020), multi-city air quality comparisons, crop correlation |
| **03** | **Linear Regression & Parameter Saving** | [`lab3/`](./lab3/) | `student_survey.csv` | Simple & Multiple Linear Regression, OLS closed-form solver, residual normality tests, Pickle weights |
| **04** | **KNN Classification & Evaluation Metrics** | [`lab4/`](./lab4/) | Breast Cancer Wisconsin | Distance metrics (Euclidean, Manhattan, Chebyshev), K-elbow search, ROC AUC, PR curves |
| **05** | **Gradient Descent Linear Regression** | [`lab5/`](./lab5/) | UCI Student Performance | Batch Gradient Descent from scratch, learning rate sweeps, loss convergence monitoring |
| **06** | **Logistic Regression vs KNN Classifier** | [`lab6/`](./lab6/) | Breast Cancer Wisconsin | Class imbalance handling, confusion matrix analysis, decision threshold tuning, F1 trade-offs |
| **07** | **Decision Tree Classification** | [`lab7/`](./lab7/) | Fisher's Iris Dataset | Information Gain (Entropy) vs Gini Impurity, tree depth pruning, decision surface visualization |
| **08** | **Categorical Naive Bayes** | [`lab8/`](./lab8/) | Weather / Play Tennis | Prior/posterior probability calculation, Laplace smoothing ($\alpha$), conditional independence |
| **09** | **Support Vector Machines & PCA / LDA** | [`lab9/`](./lab9/) | Breast Cancer & Wine | Linear & RBF kernel SVM, maximal margin hyperplanes, PCA scree plots, LDA dimensionality reduction |
| **10** | **XOR Boolean Function via Multi-Layer Perceptrons** | [`lab10/`](./lab10/) | Augmented XOR Dataset | Non-linear XOR separation, multi-framework benchmark (**Keras**, **PyTorch**, **TensorFlow** low-level) |

---

## 🚀 Continuous Internal Assessments (CIA)

- **[CIA-1: Road Accident Analysis & Severity Prediction](./cia_1/)**
  - Exploratory data analysis, economic loss modeling, missing data treatment, and correlation heatmaps on road accident statistics.
- **[CIA-3: Mission Crisis — Disaster Response Message Classification](./cia3/)**
  - Multi-output NLP pipeline classifying emergency disaster messages into actionable humanitarian response categories using TF-IDF vectorization and ensemble classifiers.

---

## 🛠️ Environment Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/K1NGS1LVER/MCA_machine_learning.git
cd MCA_machine_learning
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📄 Lab Manual Generation

To generate the complete institutional Word document (`.docx`) and rendered PDF (`.pdf`):

```bash
# Run the automated manual compiler
python generate_perfect_manual.py

# Render PDF using Pandoc and Weasyprint
pandoc Daniel_Paul_2547159_ML_Lab_Manual.docx -o Daniel_Paul_2547159_ML_Lab_Manual.pdf --pdf-engine=weasyprint
```

---

## 📜 License & Academic Integrity
All materials are developed for academic coursework in accordance with the ethical coding guidelines of the Department of Computer Science, CHRIST (Deemed to be University).
