# Predicting Credit Card Customer Churn

A data science project using advanced machine learning to proactively identify credit card customers most at risk of churning—driving smarter retention strategy and maximizing customer lifetime value.

---

## 📌 Overview

Customer churn is a costly challenge for financial institutions—each lost customer means missed revenue and increased acquisition costs.  
This project tackles the churn problem head-on by building a **robust, end-to-end ML pipeline** that predicts which customers are most likely to leave, using a publicly available Kaggle dataset with 10,000+ records.

Key achievements:
- Delivered a production-ready **XGBoost model** with a **ROC-AUC of 0.99**, outperforming baseline and tree models.
- Translated **feature importance and SHAP values** into actionable retention levers, uncovering why customers churn and what can be done about it.
- Demonstrated full ML workflow: **EDA, feature engineering, class imbalance handling (SMOTE), model selection, interpretability, and reporting**.

---

## 📊 Dataset

- **Source:** [Kaggle - Prediction of Churning Credit Card Customers](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m/data)
- **Volume:** ~10,000 customer records, 20+ features
- **Target:** `Attrition_Flag` (Attrited Customer = churned, Existing Customer = retained)
- **Features:**  
  - Demographics (age, gender, marital status, education, income)
  - Account tenure and engagement (months on book, total relationships, transaction counts)
  - Financial behavior (credit limit, revolving balance, open-to-buy, utilization ratio)

---

## ⚙️ Tools & Libraries

- **Python:** pandas, numpy, scikit-learn, XGBoost, imbalanced-learn, SHAP, matplotlib, seaborn
- **Jupyter Notebooks** for experimentation and communication
- **Modeling:** Logistic Regression, Random Forest, XGBoost
- **Interpretation:** Feature importance, SHAP value analysis

---

## 🔍 Project Workflow

1. **Data Cleaning:**  
   - Dropped irrelevant IDs, handled missing values, and ensured data integrity.
2. **EDA:**  
   - Visualized class imbalance and explored key patterns across demographic, behavioral, and financial variables.
3. **Feature Engineering:**  
   - Created new features (e.g., log-transformed amounts, engagement ratios, churn velocity).
   - Normalized high-skew attributes.
4. **Class Imbalance Handling:**  
   - Applied **SMOTE** and stratified train-test split to balance the target and improve minority class recall.
5. **Model Development:**  
   - Built and cross-validated Logistic Regression, Random Forest, and XGBoost classifiers.
   - Tuned hyperparameters using RandomizedSearchCV and stratified k-fold.
6. **Model Evaluation:**  
   - Compared models on ROC-AUC, F1, precision, recall, and confusion matrix metrics.
   - Assessed stability and robustness using test splits.
7. **Interpretability:**  
   - Used **feature importance** and **SHAP value plots** to pinpoint top churn drivers (e.g., inactivity, low transaction count, declining engagement).
   - Visualized individual and global model decisions to make results transparent for business users.
8. **Reporting & Documentation:**  
   - All analysis and results published in [GitHub](https://github.com/DigimonFrankie/Projects/tree/main/Predicting%20Credit%20Card%20Customer%20Churn%3A%20A%20Data-Driven%20Approach%20to%20Retention%20Strategy) for reproducibility and user review.

---

## 📈 Results

- **Best model:**   XGBoost (Feature Engineering + SMOTE)
- **ROC-AUC:** 0.992  
- **Recall improvement:** +10% over standard Random Forest by addressing class imbalance and feature engineering.
- **Key churn predictors (from SHAP):**
  - High months of inactivity
  - Low transaction count/frequency
  - Fewer product relationships
  - Declining transaction trends
- **Actionable insight:** Customers with dropping engagement and fewer recent transactions are at highest churn risk—these are your top retention targets.

---

## 📌 Key Takeaways

- **Technical impact:** Advanced ML and EDA delivered a high-performing, explainable churn model—proving end-to-end data science workflow expertise.
- **Business value:** The model enables **proactive, data-driven retention**—helping target offers to customers who matter most, with measurable revenue impact.
- **Interpretability:** SHAP analysis built trust and made recommendations clear for both technical and business audiences.

---

## 🧠 Future Work

- Deploy as a REST API for real-time churn scoring.
- Integrate time-series analysis to capture temporal trends in behavior.
- A/B test targeted interventions based on model outputs to measure ROI.

---

## 📚 Documentation & Notebooks

- [Optimal Models.ipynb](https://github.com/DigimonFrankie/Projects/blob/main/Predicting%20Credit%20Card%20Customer%20Churn%3A%20A%20Data-Driven%20Approach%20to%20Retention%20Strategy/Optimal%20Models.ipynb) – Final model summary and business interpretation
- [Logistic Regression (Step by step).ipynb](https://github.com/DigimonFrankie/Projects/blob/main/Predicting%20Credit%20Card%20Customer%20Churn%3A%20A%20Data-Driven%20Approach%20to%20Retention%20Strategy/Step%20by%20step/Logistic%20Regression%20(Step%20by%20step).ipynb) – Baseline modeling
- [Ensemble Trees (Step by step).ipynb](https://github.com/DigimonFrankie/Projects/blob/main/Predicting%20Credit%20Card%20Customer%20Churn%3A%20A%20Data-Driven%20Approach%20to%20Retention%20Strategy/Step%20by%20step/Emsemble%20Trees%20(Step%20by%20step).ipynb) – Random Forest & XGBoost

---

*For code, results, and detailed analysis, check out the full [GitHub repository](https://github.com/DigimonFrankie/Projects/tree/main/Predicting%20Credit%20Card%20Customer%20Churn%3A%20A%20Data-Driven%20Approach%20to%20Retention%20Strategy).*

---

## 📬 **Contact**

If you'd like to collaborate or ask questions, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/frankhzhao/) or weifu.h.zhao@gmail.com.