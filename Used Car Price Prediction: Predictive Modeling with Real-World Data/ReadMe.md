# Used Car Price Prediction

_A data science project predicting used car prices using real-world, unstructured data and advanced regression models._

---

## 📌 Overview

Accurate used car pricing is critical for dealerships and marketplaces—mispricing can mean lost profits or stale inventory.
This project tackles the price prediction challenge by building a robust, end-to-end regression pipeline using a real-world dataset reflecting volatile, COVID-era market conditions.

Key achievements:
- Engineered structured features from unstructured data (brand/model, engine, accident history, title status), handling missing values and inconsistent labeling.
- Developed and benchmarked regression models — **Multiple Linear Regression**, **Random Forest**, **XGBoost**, and **ensemble stacking** — to deliver substantial performance gains over baseline.
- Demonstrated the full machine learning workflow: **EDA**, **feature engineering**, **missing value imputation**, **model selection**, **evaluation**, and **transparent reporting**.

---

## 📊 Dataset

- **Source:** [Kaggle - Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset/data)
- **Volume:** ~4,009 records, 11 features
- **Target:** `price`
- **Features:** 
  - `Brand` & `Model`: Identify the brand or company name along with the specific model of each vehicle.
  - `Model Year`: Discover the manufacturing year of the vehicles, crucial for assessing depreciation and technology advancements.
  - `Mileage`: Obtain the mileage of each vehicle, a key indicator of wear and tear and potential maintenance requirements.
  - `Fuel Type`: Learn about the type of fuel the vehicles run on, whether it's gasoline, diesel, electric, or hybrid.
  - `Engine Type`: Understand the engine specifications, shedding light on performance and efficiency.
  - `Transmission`: Determine the transmission type, whether automatic, manual, or another variant.
  - `Exterior` & `Interior Colors`: Explore the aesthetic aspects of the vehicles, including exterior and interior color options.
  - `Accident History`: Discover whether a vehicle has a prior history of accidents or damage, crucial for informed decision-making.
  - `Clean Title`: Evaluate the availability of a clean title, which can impact the vehicle's resale value and legal status.
  - `Price`: Access the listed prices for each vehicle, aiding in price comparison and budgeting.

---

## ⚙️ Tools & Libraries

- **Python:** pandas, numpy, scikit-learn, XGBoost, etc.
- **Visualization:** matplotlib, seaborn
- **Notebook:** Jupyter Notebooks
- **Models:** Multiple Linear Regression, Random Forest, XGBoost, ensemble stacking, etc.

---

## 🔍 Project Workflow

1. **Data Cleaning & Preparation**
   - Parsed unstructured fields (brand/model, engine, etc.)
   - Extracted and standardized engine specs (hp, liters, cylinders) from messy text, handled garbage values (e.g., "617.0HP 4.4L 8 Cylinder Engine Gasoline Fuel")
   - Limited engine fields to plausible physical ranges to avoid clown data (e.g., liters < 10, cylinders < 16)
   - Handled missing values
   - Standardized categories
 
2. **Engine Spec Mapping & Imputation**
   - Built multi-key nested dictionaries for mapping engine specs by (model_year, brand, model) and sub-keys like (liters, cylinders)
   - Handled duplicate or conflicting specs by taking mode, then max if tied
   - Imputed missing specs using best-match rules:
     - Prefer same MY, otherwise look back/forward within ±3 years
     - Must match brand, model, and available engine specs (e.g., if only liters/cylinders are present, use that as lookup)
   - Filled NA for hp/liters/cylinders using engine spec map, with clear rules for tie-breaking

3. **EDA**
   - Visualized price distributions and market trends
   - Explored feature relationships and outliers

4. **Feature Engineering**
   - Created structured features (e.g., accident history as binary, normalized engine size)
   - Derived new features from raw text

5. **Model Development**
   - Built and tuned baseline and advanced models
   - Benchmarked performance with cross-validation

6. **Model Evaluation**
   - Evaluated models using RMSE, MAE, R²
   - Compared performance to baseline
   - Analyzed feature importances

7. **Reporting & Documentation**
   - Published code, results, and analysis on [GitHub](#)

---

## 📈 Results

- **Best model:** 
- **Performance:**  
  - RMSE: 
  - R²: 
  - MAE: 
  - Improvement vs. baseline: 
- **Top features influencing price:** 

---

## 📌 Key Takeaways

- **Technical impact:** 
- **Business value:** 
- **Interpretability:** 

---

## 🧠 Future Work

- 

---

## 📚 Documentation & Notebooks



---

## 📬 Contact

If you'd like to collaborate or ask questions, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/frankhzhao/) or weifu.h.zhao@gmail.com.