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
   - Evaluated models using RMSE, R²
   - Compared performance to baseline
   - Analyzed feature importances

7. **Reporting & Documentation**
   - Published code, results, and analysis on [GitHub](https://github.com/DigimonFrankie/Projects/tree/main/Used%20Car%20Price%20Prediction%3A%20Predictive%20Modeling%20with%20Real-World%20Data)

---

## 📈 Results

- **Best model:** LightGBM
- **Performance:**  
  - RMSE: 126,305.00
  - R²: 0.220
  - Improvement vs. baseline: Compared to the baseline (Elastic Net: RMSE $137,163.17, R² 0.07), the best model (LightGBM) achieved an RMSE of $126,305.00 and an R² of 0.22, representing an ~8% reduction in prediction error and substantially higher variance explained.
- **Top features influencing price:** 

![Feature Importance](/Used%20Car%20Price%20Prediction:%20Predictive%20Modeling%20with%20Real-World%20Data/src/img/feature%20importance.png)

![SHAP summary plot](/Used%20Car%20Price%20Prediction:%20Predictive%20Modeling%20with%20Real-World%20Data/src/img/shap.png)
---

## 📌 Key Takeaways

- **Technical impact:**  
    Developed a fully automated regression pipeline using advanced tree-based algorithms (LightGBM, XGBoost, CatBoost) and linear models, enabling efficient hyperparameter tuning, robust feature engineering, and reproducible model selection. Improved model performance on the validation set by 8.5% RMSE reduction versus baseline, with modular Python code for easy extension.

- **Business value:**  
    Identified the most influential factors driving used car prices—such as horsepower, mileage, model year, and brand—empowering data-driven pricing strategies. The model enables stakeholders to better understand price drivers, detect under- or over-valued listings, and optimize inventory or pricing policies based on explainable predictions.

- **Interpretability:**  
    Leveraged SHAP analysis to provide transparent explanations of model predictions, showing how key features like horsepower, mileage, and car age affect price estimates. This interpretability builds trust with business users and ensures that the model’s logic aligns with domain knowledge and industry expectations.

---

## 🧠 Future Work

- **Data sample size:**  
    Increase dataset size by collecting more historical transactions and integrating external sources. A larger, more diverse dataset will improve model generalizability and predictive performance, especially for underrepresented car types.

- **Segmented modeling:**  
    Build separate models for different vehicle segments—such as exotic, premium, and mass-market cars—to better capture pricing dynamics unique to each category. This could involve stratifying the data by brand or model tier before training.

- **Model enhancement:**  
    Experiment with additional advanced algorithms (e.g., neural networks, stacked ensembles) and leverage model ensembling for further performance gains.

- **Feature enrichment:**  
    Incorporate new data sources (market demand, regional trends, seller/buyer ratings) or extract features from car images using computer vision for richer context.

- **Handling outliers and rare categories:**  
    Develop specialized models or preprocessing strategies for extreme price values and rare brands/models to improve robustness.

- **Temporal modeling:**  
    Integrate time-series techniques to better capture market seasonality and price trends over time.

- **Model deployment:**  
    Deploy the best model as an API or dashboard for real-time pricing support to end-users or business stakeholders.

- **User feedback loop:**  
    Incorporate feedback mechanisms to learn from actual sale prices, continuously retraining and updating the model for sustained accuracy.

---

## 📚 Documentation & Notebooks



---

## 📬 Contact

If you'd like to collaborate or ask questions, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/frankzhaods/) or frank.zhao.ds@outlook.com.