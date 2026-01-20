# Loan Default Risk Prediction and Approval Policy Evaluation

_A data science project simulating loan approval policy changes and predicting loan default risk using a real-world lending dataset and machine learning._

---

## 📌 Overview

Lenders constantly balance approval rates and default risk when setting credit policy. Small changes in credit score thresholds can have massive effects on profit, risk, and customer access. This project evaluates the impact of changing approval policies through simulated A/B testing and builds robust predictive models for loan default, leveraging a large, real-world lending dataset.

Key achievements:
- Simulated old vs. new credit policy scenarios using applicant `Credit_Score` and compared their effect on approval rates, default rates (`Status`), and projected revenue.
- Performed comprehensive **EDA** to uncover risk drivers, segment impacts, and data quality issues.
- Built and benchmarked machine learning models (**Logistic Regression**, **Decision Trees**) for default prediction, including feature engineering and model interpretation.
- Quantified business tradeoffs and provided actionable, data-driven recommendations for optimizing loan portfolio risk and revenue.

---

## 📊 Dataset

- **Source:** [Kaggle Loan Default Dataset](https://www.kaggle.com/datasets/yasserh/loan-default-dataset?utm_source=chatgpt.com)
- **Volume:** 148,670 records, 34 features
- **Target:** `Status` (loan default outcome)
- **Key features:**
  - `Credit_Score`, `loan_amount`, `income`, `rate_of_interest`, `LTV`, `dtir1`, applicant demographics, property and loan details

---

## 🧾 Feature Dictionary

| Column                     | Description                                                                                       |
|----------------------------|---------------------------------------------------------------------------------------------------|
| **ID**                     | Unique loan application ID                                                                        |
| **year**                   | Year of the loan or application (e.g., 2019)                                                     |
| **loan_limit**             | Loan limit category (`cf` = conforming; blank = missing/unknown)                                  |
| **Gender**                 | Borrower's gender (`Male`, `Female`, `Joint`, `Sex Not Available`)                               |
| **approv_in_adv**          | Approval in advance status (`pre`, `nopre`)                                                      |
| **loan_type**              | Loan type (`type1`, `type2`, etc.; dataset-specific codes)                                       |
| **loan_purpose**           | Loan purpose (`p1`, `p3`, `p4`, etc.; coded—likely purchase/refinance/other)                     |
| **Credit_Worthiness**      | Borrower creditworthiness tier (`l1`, `l2`, etc.; coded)                                         |
| **open_credit**            | Number of open credit lines (`nopc` = none, otherwise numeric or coded)                          |
| **business_or_commercial** | Business or commercial loan indicator (`b/c` = yes, `nob/c` = no)                                |
| **loan_amount**            | Loan amount requested or granted                                                                 |
| **rate_of_interest**       | Interest rate assigned to the loan (can be blank)                                                |
| **Interest_rate_spread**   | Spread above benchmark/reference rate (may be negative, zero, or blank)                          |
| **Upfront_charges**        | Origination or upfront fees (can be zero or blank)                                               |
| **term**                   | Loan term in months (`360`, `180`, `300`, etc.)                                                  |
| **Neg_ammortization**      | Negative amortization allowed (`neg_amm`, `not_neg`)                                             |
| **interest_only**          | Interest-only payment allowed (`not_int` = no)                                                   |
| **lump_sum_payment**       | Lump sum repayment allowed (`lpsm`, `not_lpsm`)                                                  |
| **property_value**         | Appraised value of the financed property                                                         |
| **construction_type**      | Construction type (`sb` = stick-built; dataset code)                                             |
| **occupancy_type**         | Occupancy type (`pr` = primary residence, `sr` = second residence, etc.)                        |
| **Secured_by**             | Collateral type (`home`, `land`, etc.)                                                           |
| **total_units**            | Number of housing units (`1U` = one unit, etc.)                                                  |
| **income**                 | Borrower's income (may be blank)                                                                 |
| **credit_type**            | Type/source of credit report (`EXP`, `CIB`, `CRIF`, etc.)                                       |
| **Credit_Score**           | Borrower's credit score (numeric)                                                                |
| **co-applicant_credit_type**| Co-applicant's credit report type (`EXP`, `CIB`, etc.; may be blank)                            |
| **age**                    | Age group (`25-34`, `35-44`, `45-54`, `55-64`, `65-74`, `>74`)                                  |
| **submission_of_application**| Application submission channel (`to_inst`, `not_inst`)                                         |
| **LTV**                    | Loan-to-value ratio (%)                                                                          |
| **Region**                 | Geographic region (`North`, `south`, `central`)                                                  |
| **Security_Type**          | Security/collateral type (`direct`, other codes)                                                 |
| **Status**                 | Target variable: Loan default indicator (`1` = defaulted, `0` = not defaulted)                   |
| **dtir1**                  | Debt-to-income ratio (DTI, numeric)                                                              |
---

## ⚙️ Tools & Libraries

- **Python:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Modeling:** Logistic Regression, Decision Trees (other models as extensions)
- **Notebook:** Jupyter Notebooks

---

## 🔍 Project Workflow

1. **Data Cleaning & Preparation**
   - Checked for missing values, outliers, and inconsistencies (`rate_of_interest`, `Upfront_charges`, `dtir1`)
   - Transformed categorical columns (`age` to ordinal, one-hot encoding for `loan_type`, `loan_purpose`, etc.)
   - Standardized and validated numeric ranges for financial features

2. **Exploratory Data Analysis (EDA)**
   - Visualized distributions and relationships (e.g., default rate vs. credit score, income, DTI)
   - Analyzed demographic and regional trends
   - Explored segment-specific risk factors and policy impacts

3. **Customer Segmentation via Clustering**
   - Used Elbow and Silhouette methods to select the optimal number of customer groups.
   - Applied K-Means to assign each record to a cluster based on engineered numerical and encoded categorical features.
   - Calculated and compared the default rates within each cluster to reveal risk patterns.
   - Used PCA to visualize and interpret the primary drivers behind cluster formation.
   - Summarized key characteristics (e.g., financial, demographic, behavioral) for each cluster to enable actionable business segmentation.
   - Linked cluster findings to potential strategies for targeted credit policy, product offers, or risk mitigation.

4. **Machine Learning Modeling**
   - Built predictive models (Logistic Regression, Decision Trees) using engineered features
   - Evaluated model performance (ROC-AUC, confusion matrix, feature importance)
   - Interpreted key predictors and their effect on default probability

5. **Business Impact Analysis**
   - Quantified revenue and loss tradeoffs under different approval scenarios
   - Visualized business metrics and policy outcomes for strategic decision-making
   - Recommended optimal policy threshold and approval strategy

6. **Reporting & Documentation**
   - Compiled code, visualizations, and findings in organized Jupyter notebooks and Markdown reports

---

## 📈 Results

- **Customer Segmentation:**  
    - [To be filled in: Identified distinct customer groups using clustering, compared default rates and financial profiles across segments, and derived interpretable risk patterns to support targeted credit strategy.]
- **Machine Learning Model:**  
    - [To be filled in: Model performance metrics—ROC-AUC, confusion matrix, top predictive features.]
- **Business Impact:**  
    - [To be filled in: Quantified tradeoffs, recommended policy thresholds, and potential portfolio improvements.]

---

## 📌 Key Takeaways

- **Demonstrated the impact** of credit policy changes on loan approvals, defaults, and revenue using real-world data.
- **Developed interpretable ML models** for default prediction, providing actionable insights for risk management.
- **Delivered a fully documented, end-to-end analysis**—from EDA to policy simulation, modeling, and business recommendations.

---

## 🧠 Future Work

- Add more advanced models and ensemble techniques
- Expand feature engineering (interaction terms, external data sources)
- Fairness analysis across demographic groups
- Build an interactive dashboard for real-time policy simulation
- Deploy best model as a simple API or tool

---

## 📚 Documentation & Notebooks

- `Data_Cleaning_and_EDA.ipynb` – Data loading, cleaning, and initial analysis
- `Policy_Simulation.ipynb` – Policy group assignment, A/B testing, revenue analysis
- `ML_Modeling.ipynb` – Model development, validation, interpretation
- `reports/` – Final summary report, presentation slides, and Markdown documentation

---

## **Feature Data Dictionary**

| Column         | Description                        |
|----------------|------------------------------------|
| ...            | ... (see above)                    |


---

## 📬 Contact

Questions or collaboration?  
[LinkedIn](https://www.linkedin.com/in/frankzhaods/) | frank.zhao.ds@outlook.com