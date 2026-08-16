# Mission Crisis — Disaster Response Message Classification

## Problem Statement
During natural disasters, emergency responders are overwhelmed with thousands of incoming messages (social media, SMS, news). Manually triaging these is slow and error-prone. This project builds an ML pipeline to **automatically classify and prioritise emergency-help requests** into humanitarian categories (water, food, medical help, search & rescue, etc.), enabling faster response and saving lives.

## Dataset
- **Source:** [Figure Eight / Appen Disaster Response Messages](https://www.kaggle.com/datasets/thedevastator/disaster-response-messages) 
- **Size:** ~30,000 real messages from Haiti earthquake (2010), Chile earthquake (2010), Pakistan floods (2010), Hurricane Sandy (2012)
- **Unit of analysis:** Individual emergency message
- **Target:** Multi-label humanitarian categories (36 binary labels) — we focus on `request` (binary: is this message a request for help?)
- **License:** Public domain / open dataset

## Beneficiaries
- Disaster response organisations (FEMA, Red Cross, UNDP)
- Local emergency management agencies in disaster-prone regions
- Affected populations who need faster aid routing

## Why Machine Learning?
- Volume of messages during disasters exceeds human triage capacity
- Patterns in language can reliably signal urgency and category
- Automated classification enables real-time prioritisation

## Project Structure
```
cia3/
├── README.md
├── requirements.txt
├── mission_crisis_pipeline.ipynb  # Complete Jupyter notebook
└── data/                          # Auto-downloaded by the notebook
```

## How to Run

### 1. Activate the parent directory's venv
```bash
source ../venv/bin/activate
```

### 2. Ensure Kaggle credentials
Set up `~/.kaggle/kaggle.json` or environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`.

### 3. Run the notebook
```bash
jupyter notebook mission_crisis_pipeline.ipynb
```

The notebook auto-downloads the dataset from Kaggle via `kagglehub`.

## Models Implemented
| Model | Type |
|-------|------|
| Logistic Regression | Baseline |
| Random Forest | Bagging ensemble |
| XGBoost | Boosting ensemble |
| EBM (Explainable Boosting Machine) | Glass-box model |
| Stacking Classifier | Heterogeneous stacking |
| Voting Classifier | Heterogeneous voting |

## Explainability
- **SHAP** — global & local explanations for XGBoost
- **EBM** — inherently interpretable glass-box model with built-in feature importance

## Ethics Statement
- Dataset contains no personally identifiable information
- Model should augment, not replace, human decision-making in emergency response
- False negatives (missing a real emergency) are costlier than false positives
- Model performance may degrade on disasters unlike training data (domain shift)
- Bias risk: English-language messages over-represented; non-English speakers may be under-served
- Deployment requires human oversight and regular retraining

## Citation
Appen / Figure Eight. "Multilingual Disaster Response Messages." Available at: https://www.kaggle.com/datasets/thedevastator/disaster-response-messages

---
*Built for MCA 521-4 Machine Learning — ML for Social Good Challenge*
