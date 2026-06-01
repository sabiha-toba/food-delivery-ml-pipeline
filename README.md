# Food Delivery Logistics & Predictive Machine Learning Pipeline

An end-to-end machine learning pipeline built to optimise urban food delivery operations by mapping delivery data distributions, predicting order duration times, and classifying customer complaint risks under strict data-leakage constraints.

## 📂 Project Directory Structure
* **`EDA/`**: Analysis of structural data missingness, feature correlations, and a coarse 2-cluster segmentation profile using K-Means and Principal Component Analysis (PCA).
* **`Predictive_Models/`**: Supervised learning pipelines containing tuned regression and imbalanced classification models.

## ⚙️ Core Engineering Pipeline
* **Exploratory Data Analysis:** Profiled right-skewed operational metrics and isolated structural missingness (75.6% missingness in complaint fields for non-complaint orders) to prevent target leakage.
* **Feature Engineering Layer:** Implemented cyclical sine/cosine encodings for time-of-day/day-of-week variables alongside custom supply-demand ratio features (busy ratios, outstanding orders per partner).
* **Fulfillment Duration Forecasting (Regression):** Built and evaluated an ExtraTrees Regressor tuned via `RandomizedSearchCV` ($n_{\text{estimators}}=600$, $\text{max\_depth}=12$, $\text{min\_samples\_leaf}=2$, $\text{max\_features}=0.8$). Successfully captured non-linear feature interactions to achieve a holdout MAE of 12.96 minutes, outperforming baseline linear and dummy models.
* **Risk Screening (Imbalanced Classification):** Handled a moderately imbalanced 24.4% complaint class using balanced class weights inside an ExtraTrees Classifier. Tuned the hyperparameter structure ($n_{\text{estimators}}=300$, $\text{max\_depth}=20$, $\text{min\_samples\_leaf}=10$, $\text{max\_features}=\text{'sqrt'}$) with a fixed decision threshold at 0.43 to hit an operational holdout PR-AUC of 0.498 and a strong 0.673 recall rate.

## 🛠️ Technology Stack
* **Language:** Python
* **ML Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
* **Algorithms:** K-Means, Principal Component Analysis (PCA), ExtraTrees Regressor/Classifier, Ridge Regression

## 📊 Operational Contribution
Translated statistical outputs into functional risk-triage matrices. The regression pipeline categorizes delivery time frames into three operationally actionable dispatch risk tiers, while the classification pipeline acts as an early-warning system to trigger proactive customer service recovery loops right at the moment of order creation.
