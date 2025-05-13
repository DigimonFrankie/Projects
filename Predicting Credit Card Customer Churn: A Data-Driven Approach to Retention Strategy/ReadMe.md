# Predicting Credit Card Customer Churn

A machine learning project to identify customers at risk of churning based on demographic, behavioral, and financial data.

---

## 📌 Overview

Customer churn is a major concern for financial institutions, especially credit card companies. 

In this project, I developed a machine learning pipeline to predict customer churn for a credit card provider using a publicly available dataset from Kaggle. The dataset includes demographic, behavioral, and transactional attributes for over 10,000 customers. 

I performed **exploratory data analysis (EDA)**, handled class imbalance, engineered meaningful features, and trained multiple classification models including **logistic regression**, **random forest**, and **XGBoost**. 

Model performance was evaluated using **ROC-AUC**, **precision-recall metrics**, and **confusion matrices**. Feature importance analysis helped identify key drivers of churn, offering actionable insights for retention strategies. 

This project demonstrates my ability to frame a business problem, apply ML techniques, and communicate insights effectively.

---

## 📊 Dataset

- Source: [Kaggle - Prediction of Churning Credit Card Customers](https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m/data)
- Total records: ~10,000 customers
- Target variable: `Attrition_Flag` (Yes = churned, No = retained)
- Features: Demographics, engagement metrics, credit usage, transaction data

---

## ⚙️ Tools & Libraries

- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter Notebooks
- Machine Learning models: Logistic Regression, Random Forest, XGBoost
- Evaluation: Confusion Matrix, ROC-AUC, Precision-Recall

---

## 🔍 Project Workflow

1. **Data Cleaning**  
   - Remove irrelevant fields (e.g., `CLIENTNUM`, pre-computed Naive Bayes column)
   - Handle missing values
2. **Exploratory Data Analysis (EDA)**  
   - Visualize class imbalance  
   - Analyze relationships between features and churn
3. **Feature Engineering**  
   - Derived variables (e.g., engagement ratio, activity flags)
4. **Modeling**  
   - Baseline: Logistic Regression  
   - Tree-based models: Random Forest, XGBoost
5. **Evaluation**  
   - ROC-AUC, confusion matrix, classification report
   - Feature importance interpretation
6. **Insights**  
   - What behaviors indicate risk?
   - What actions can a business take?

---

## 📈 Results

- **Best model**:   
- **ROC-AUC Score**: 
- **Top churn predictors**:
  

---

## 📌 Key Takeaways

<!-- - Class imbalance was handled using SMOTE and stratified sampling.
- Feature importance revealed actionable retention levers.
- The model could help prioritize which customers should receive retention offers. -->

---

## 🧠 Future Work

<!-- - Deploy model as a REST API
- Incorporate time-series data for temporal trends
- Test interventions based on predictions -->

---

## 📬 Contact

If you'd like to collaborate or ask questions, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/frankhzhao/) or weifu.h.zhao@gmail.com.