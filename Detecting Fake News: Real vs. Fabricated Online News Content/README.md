# Detecting Fake News: Real vs. Fabricated Online News Content

A data science project using modern NLP and machine learning to separate the facts from the detecting fake news stories with end-to-end classification, interpretability, and a modular workflow.

---

## 📌 Overview

Fake news and misinformation are everywhere, poisoning public discourse and making it hard to trust what you read.  
This project tackles the fake news epidemic by building a **robust, modular ML pipeline** to classify news headlines and articles as real or fake, using a large public dataset of labeled media.

Key achievements:
- Delivered a production-ready **Logistic Regression baseline**, **SVM**, **Naive Bayes**, and **deep learning models (LSTM/BERT)** for state-of-the-art detection accuracy.
- Modularized all text preprocessing in a reusable Python module for consistent, efficient data cleaning.
- Mapped out feature importance and model explanations to reveal what drives “fake” vs. “real” decisions.
- Demonstrated a complete workflow: **EDA, preprocessing, feature engineering, model selection, interpretability, and clear reporting**.

---

## 📊 Dataset

- **Source:** [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Volume:** 40,000+ labeled news articles (title, text, subject, date)
- **Target:** `label` (FAKE or REAL)
- **Features:**
  - `title`: Headline text
  - `text`: Full news article
  - `subject`: News category
  - `date`: Date published

---

## ⚙️ Tools & Libraries

- **Python:** pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, wordcloud, TensorFlow/Keras, HuggingFace Transformers (for BERT)
- **Preprocessing module:** All text cleaning/tokenization in `preprocessing.py`
- **Jupyter Notebooks** for exploration and modeling
- **Modeling:** Logistic Regression, SVM, Naive Bayes, LSTM, BERT
- **Interpretation:** Feature importance, model coefficients, LIME/SHAP for deep learning

---

## 🔍 Project Workflow

1. **Data Cleaning & Preprocessing:**  
   - Used a custom `preprocessing.py` module for text normalization, tokenization, stopword removal, and lemmatization.
   - Dropped irrelevant columns, handled missing values, and standardized text.
2. **EDA:**  
   - Explored class balance, headline/article length, common words, and n-gram patterns in fake vs. real news.
3. **Feature Engineering:**  
   - Created features from both titles and full article text.
   - Vectorized text using TF-IDF and CountVectorizer.
   - Explored subject/category as an auxiliary feature.
4. **Model Development:**  
   - Built and cross-validated Logistic Regression, SVM, and Naive Bayes classifiers as strong baselines.
   - Trained deep learning models (LSTM and BERT) for maximum accuracy on article text.
5. **Model Evaluation:**  
   - Compared models on accuracy, precision, recall, F1, and confusion matrix.
   - Assessed robustness with train-test splits and cross-validation.
6. **Interpretability:**  
   - Analyzed model coefficients and feature importances.
   - Used LIME/SHAP to explain predictions for selected articles.
   - Visualized top drivers of “fake” and “real” predictions.
7. **Reporting & Documentation:**  
   - All code and results are published in [GitHub](#) for full transparency and reproducibility.

---

## 📈 Results

- **Best model:** 
- **Accuracy:** 
- **Precision/Recall:** 
- **Key predictors (from interpretability):**

- **Actionable insight:**  


---

## 📌 Key Takeaways



---

## 🧠 Future Work


---

## 📚 Documentation & Notebooks

- [EDA & Preprocessing.ipynb](#) – Data cleaning, exploratory data analysis, and visualization  - 🚧 in progress
- [Logistic Regression.ipynb](#) – Building and evaluating a logistic regression baseline  - 🚧 in progress
- [SVM.ipynb](#) – Training and assessing a Support Vector Machine classifier  - 📋 to-do
- [Naive Bayes.ipynb](#) – Implementing and interpreting a Naive Bayes model  - 📋 to-do
- [LSTM.ipynb](#) – Applying LSTM neural networks for sequential text classification  - 📋 to-do
- [BERT Deep Learning.ipynb](#) – Fine-tuning and testing a transformer-based BERT model  - 📋 to-do
- [Interpretability & Reporting.ipynb](#) – Explaining model decisions, visualizing feature importances, and summarizing results  - 📋 to-do

---

*For code, results, and full analysis, check out the complete [GitHub repository](#).*

---

## 📬 **Contact**

If you'd like to collaborate or have questions, reach out via [LinkedIn](https://www.linkedin.com/in/frankhzhao/) or weifu.h.zhao@gmail.com.

---