#  Credit Card Fraud Detection Analytics

An End-to-End Machine Learning Pipeline to detect fraudulent financial transactions in real-time. This project addresses the extreme class imbalance problem inherent in financial datasets and deploys a predictive model via an interactive Streamlit web dashboard.

---

##  Dashboard Preview
Here is how the live dashboard looks with interactive charts and real-time risk scoring:

![Fraud Detection Dashboard](dashboard.png)

---

##  Project Overview
Financial fraud costs billions of dollars annually. This project implements a robust data engineering and machine learning workflow to identify fraudulent activities with high precision and recall. 

### Key Features:
- **Advanced Imbalance Handling:** Utilized SMOTE (Synthetic Minority Over-sampling Technique) to tackle extreme class imbalance (only 0.17% fraud instances).
- **Production-Grade Algorithm:** Trained using optimized **XGBoost** classifier.
- **Interactive Dashboard:** Deployed a functional UI using **Streamlit** for real-time transaction scoring.
- **Robust Evaluation:** Focused on **Precision-Recall AUC** and **ROC-AUC** metrics rather than simple accuracy.

---

##  Dataset
The dataset used is the **Kaggle Credit Card Fraud Detection Dataset**. It contains features which are a result of a PCA transformation due to confidentiality issues.
- **Total Transactions:** 284,807
- **Fraudulent Transactions:** 492 (0.172%)

---

##  Tech Stack & Architecture
- **Language:** Python 3.x
- **Libraries:** Scikit-Learn, XGBoost, Imbalanced-Learn, Pandas, NumPy
- **Deployment:** Streamlit

---

##  How to Run Locally

### 1. Clone the repository
```bash
git clone [https://github.com/akanksha1066/Fraud-Detection-Analytics.git](https://github.com/akanksha1066/Fraud-Detection-Analytics.git)
cd Fraud-Detection-Analytics
