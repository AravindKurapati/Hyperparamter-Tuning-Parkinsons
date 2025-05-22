# Hyperparameter Tuning for Early Diagnosis of Parkinson’s Disease Using Big Data

This project applies supervised machine learning to **detect Parkinson's Disease** based on vocal biomarkers. It explores how **hyperparameter tuning** improves classification performance and compares the effectiveness of different ML models using a publicly available biomedical dataset.

---

##  Objective

The primary goal is to build a model that can classify whether a patient has Parkinson’s disease based on voice frequency and signal features. We apply **GridSearchCV** to identify the best hyperparameters for various models.

---

##  Dataset

- 📦 Source: [UCI Parkinson’s Disease Data Set](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- 195 samples (147 Parkinson's, 48 healthy)
- 23 voice features extracted from sustained phonation recordings:
  - MDVP: Frequency, jitter, shimmer
  - HNR, DFA, RPDE
- Target: `status` (1 = Parkinson's, 0 = Healthy)

---

## Preprocessing Steps

- Checked for missing values
- Normalized features with **StandardScaler**
- Checked feature correlations (heatmap)
- Separated features/labels and split into train/test

---

## Models Evaluated

| Model              | Tuning Performed         |
|-------------------|--------------------------|
| Logistic Regression | No (baseline)           |
| Support Vector Machine (SVM) |  C, kernel, gamma |
| K-Nearest Neighbors (KNN) |  n_neighbors, weights, metric |
| Decision Tree      |  max_depth, min_samples_split |
| Random Forest      |  n_estimators, max_depth |

Used **GridSearchCV** with 5-fold cross-validation for tuning.

---

## Results

| Model              | Accuracy | Best Parameters |
|-------------------|----------|-----------------|
| Logistic Regression | 88.46%  | -               |
| SVM (RBF)           | **94.87%** | C=1000, gamma=0.001 |
| Random Forest       | 91.02%  | n=50, max_depth=5 |
| KNN                 | 87.17%  | k=3, metric='manhattan' |
| Decision Tree       | 85.89%  | max_depth=5      |

 **SVM** with RBF kernel and tuned hyperparameters achieved the highest accuracy and best generalization.

---


##  Key Takeaways

- Hyperparameter tuning significantly boosts performance, especially for SVM and Random Forest.
- Simple classifiers like KNN and Logistic Regression work reasonably well but plateau early.
- The dataset is slightly imbalanced but tuning improves sensitivity to the minority class (healthy patients).

---

## Tools & Libraries

- Python (NumPy, Pandas, Scikit-learn)
- Matplotlib, Seaborn for plots
- GridSearchCV for model optimization

