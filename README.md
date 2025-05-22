
# 🕵️‍♀️ Fake Review Detection using Supervised Machine Learning

This project aims to detect **fake product reviews** using supervised machine learning techniques. The motivation stems from the increasing presence of deceptive reviews that mislead consumers and damage trust in online platforms.

---

## 📌 Objectives
- Identify and understand characteristics of fake reviews.
- Use **heuristic labeling** with real and AI-generated (T5) reviews.
- Build and compare multiple supervised ML models.
- Analyze model predictions to understand deceptive review patterns.

---

## 🧾 Dataset
- **Source**: [Amazon Product Review Dataset (Electronics)](https://snap.stanford.edu/data/amazon/productGraph/)
- **Composition**:
  - Real reviews (OG) from dataset
  - AI-generated fake reviews (CG) using T5-small
  - Balanced dataset (~50% OG, ~50% CG)
  - Final cleaned file: `cleandata.csv`

---

## 🧹 Preprocessing
- Removed irrelevant fields (IDs, rating, etc.)
- Text cleaned (lowercase, stopwords, punctuation, stemming)
- Applied **TF-IDF vectorization**
- Data split: 80% train / 20% test

---

## 🤖 Models & Results

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 87%      |
| Decision Tree      | 83%      |
| Random Forest      | 88%      |
| SVM (Linear)       | 88%      |
| Naive Bayes        | 69%      |
| KNN                | 68%      |
| **XGBoost**        | **90.5%** ✅ |

- **Best Model**: XGBoost
  - Precision: 90.7%
  - Recall: 90.6%
  - F1-Score: 90.6%
  - ROC-AUC: 0.97

---

## 🔍 Key Insights
- CG reviews tend to be shorter than OG reviews.
- Important TF-IDF features include: `"product"`, `"recommend"`, `"fake"`.
- XGBoost performed best across all evaluation metrics.

---

## 🚀 Future Enhancements
- Use verified fake review datasets.
- Include emoji and multi-language reviews.
- Try deep learning models for further performance boost.

---

## 📚 References
- [IEEE Access papers on fake review and deepfake detection]
- [T5 Language Model – Hugging Face]
