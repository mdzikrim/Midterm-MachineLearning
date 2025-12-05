# UTS Machine Learning - Fraud Detection, Year Prediction, and Customer Segmentation

## 👤 Student Identification

- **Name:** Muhammad Dzikri Muqimulhaq
- **NIM:** 1103220147
- **Class:** Machine Learning TK 46-GAB

---

## 📋 Purpose of Repository

This repository contains three comprehensive machine learning projects developed as part of the Midterm Examination (UTS) for the Machine Learning course. The projects demonstrate proficiency in:

1. **Classification** - Fraud Detection in Financial Transactions
2. **Regression** - Year Prediction from Audio Features
3. **Clustering** - Customer Segmentation for Credit Card Users

Each project showcases the complete machine learning pipeline including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, hyperparameter tuning, and comprehensive evaluation.

---

## 🚀 Project Overview

### 1. **Fraud Detection (uts_transaction.ipynb)** 🔐

**Problem Statement:**  
Detect fraudulent transactions in a highly imbalanced financial transaction dataset using binary classification techniques.

**Dataset:**
- **Train Set:** 590,540 transactions with 394 features
- **Test Set:** 506,691 transactions with 393 features
- **Target Variable:** `isFraud` (0 = Normal, 1 = Fraud)
- **Class Imbalance:** ~96.5% Normal, ~3.5% Fraud (ratio 28:1)

**Key Techniques:**
- **Missing Value Handling:** Dropped columns with >90% missing values, filled numeric features with median, categorical with 'Unknown'
- **Feature Engineering:**
  - Transaction Amount: Log transformation, decimal extraction, round amount detection
  - Time Features: Day, hour, night time flag, weekend flag
- **Label Encoding:** Converted categorical variables to numerical format
- **SMOTE (Synthetic Minority Over-sampling Technique):** Addressed class imbalance with sampling_strategy=0.3
- **Models:** Random Forest, XGBoost
- **Hyperparameter Tuning:** GridSearchCV with cross-validation

**Models & Performance:**

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Random Forest | 0.9743 | 0.9250 | 0.8850 | 0.9045 |
| **XGBoost (Best)** | **0.9905** | **0.9650** | **0.9414** | **0.9530** |

**Key Results:**
- ✅ **Best Model:** XGBoost with ROC-AUC of **0.9905** (99.05%)
- ✅ Excellent precision (96.50%) minimizes false positives
- ✅ High recall (94.14%) ensures most frauds are detected
- ✅ Feature importance analysis revealed most influential fraud indicators

**Insights:**
- Transaction amount features and time-based features are crucial for fraud detection
- XGBoost significantly outperforms Random Forest due to its gradient boosting mechanism
- SMOTE effectively balanced the dataset, improving minority class detection

---

### 2. **Year Prediction from Audio Features (uts_regresi.ipynb)** 🎵

**Problem Statement:**  
Predict the release year of songs (1922-2011) based on 90 audio features extracted from the Million Song Dataset using regression techniques.

**Dataset:**
- **Total Samples:** 515,345 songs
- **Features:** 90 audio characteristics (timbre, pitch, loudness, etc.)
- **Target Variable:** `year` (1922-2011)
- **Dataset Size:** ~358 MB

**Key Techniques:**
- **Data Exploration:** Analyzed year distribution, identified outliers (5.76% songs from 1922-1975)
- **No Missing Values:** Dataset is complete
- **Feature Scaling:** StandardScaler for normalization
- **Train-Test Split:** 80% training (412,276 samples), 20% testing (103,069 samples)
- **Models:** Linear Regression, Random Forest, XGBoost
- **Hyperparameter Tuning:** Manual configuration testing
- **Learning Curve Analysis:** Model performance vs. dataset size

**Models & Performance:**

| Model | RMSE (years) | MAE (years) | R² Score |
|-------|--------------|-------------|----------|
| Linear Regression | ~9.10 | ~6.50 | 0.30 |
| Random Forest | ~8.95 | ~6.28 | 0.33 |
| **XGBoost (Best)** | **8.83** | **6.16** | **0.3445** |

**Hyperparameter Tuning Results:**

| Config | max_depth | learning_rate | n_estimators | R² Score | RMSE |
|--------|-----------|---------------|--------------|----------|------|
| Config 1 (Default) | 8 | 0.1 | 150 | 0.3445 | 8.8327 |
| **Config 2 (Best)** | **10** | **0.05** | **200** | **0.3553** | **8.7595** |
| Config 3 | 6 | 0.15 | 100 | 0.3265 | 8.9531 |

**Key Results:**
- ✅ **Best Model:** XGBoost with R² Score of **0.3553** after tuning
- ✅ Average prediction error: **±6.16 years** (MAE)
- ✅ **Prediction Accuracy:**
  - 56.68% of predictions within ±5 years
  - 82.01% of predictions within ±10 years
- ✅ Model explains 35.53% of variance in release years

**Insights:**
- Audio features have moderate predictive power for release year (R² ~0.35)
- Modern songs (1990s-2010s) are easier to predict than older songs (1920s-1970s)
- Feature importance analysis shows certain audio characteristics (specific features) are strongly correlated with era
- The task is inherently challenging due to musical diversity within decades

---

### 3. **Customer Segmentation (uts_clustering2.ipynb)** 💳

**Problem Statement:**  
Segment credit card customers into distinct groups based on their transaction behavior and credit usage patterns using unsupervised clustering techniques.

**Dataset:**
- **Total Customers:** 8,950
- **Features:** 17 behavioral features + 1 customer ID
  - BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS
  - Frequency metrics: PURCHASES_FREQUENCY, CASH_ADVANCE_FREQUENCY
  - Transaction counts: PURCHASES_TRX, CASH_ADVANCE_TRX
  - TENURE, MINIMUM_PAYMENTS, PRC_FULL_PAYMENT
- **Missing Values:** MINIMUM_PAYMENTS (313 values, 3.5%), CREDIT_LIMIT (1 value, 0.01%)

**Key Techniques:**
- **Missing Value Handling:** Filled with median (robust to outliers)
- **Outlier Detection:** IQR method - outliers retained as they represent valuable segments
- **Feature Scaling:** StandardScaler (critical for distance-based algorithms)
- **Optimal K Selection:**
  - Elbow Method (visual identification)
  - Silhouette Analysis (quantitative validation)
- **Algorithms:** K-Means, Hierarchical Clustering (Ward linkage), DBSCAN
- **Dimensionality Reduction:** PCA for 2D visualization

**Optimal K Selection Results:**

| K | Inertia | Silhouette Score | Interpretation |
|---|---------|------------------|----------------|
| 2 | 129,820 | 0.2455 | Poor structure |
| **3** | **111,975** | **0.2510** | **Selected (highest)** |
| 4 | 99,150 | 0.2380 | Poor structure |
| 5 | 89,420 | 0.2295 | Poor structure |

**K-Means Clustering Results (K=3):**

| Cluster | Size | Percentage | Key Characteristics |
|---------|------|------------|---------------------|
| **Cluster 0** | 1,275 | 14.25% | **High Spenders** - High purchases ($4,187), High frequency (0.95), Premium customers |
| **Cluster 1** | 6,114 | 68.31% | **Regular Users** - Moderate balance ($808), Low-medium purchases ($496), Majority segment |
| **Cluster 2** | 1,561 | 17.44% | **Cash Advance Users** - High cash advance ($3,917), Low purchases ($389), High balance ($4,024) |

**Detailed Cluster Profiles:**

**Cluster 0 - Premium Active Buyers (14.25%)**
- Average Balance: $2,182.35
- Average Purchases: $4,187.02
- Average Cash Advance: $449.75
- Credit Limit: $7,642.78
- Purchase Frequency: 0.95 (very high)
- **Profile:** Premium customers who actively use their cards for purchases, high engagement

**Cluster 1 - Regular Low-Activity Users (68.31%)**
- Average Balance: $807.72
- Average Purchases: $496.06
- Average Cash Advance: $339.00
- Credit Limit: $3,267.02
- Purchase Frequency: 0.46 (moderate)
- **Profile:** Typical customers with moderate card usage, represents the majority

**Cluster 2 - Cash-Dependent High-Balance (17.44%)**
- Average Balance: $4,023.79
- Average Purchases: $389.05
- Average Cash Advance: $3,917.25
- Credit Limit: $6,729.47
- Purchase Frequency: 0.23 (low)
- **Profile:** Customers who primarily use cash advance feature, high balance, potential risk group

**Clustering Performance:**

| Method | Silhouette Score | Notes |
|--------|------------------|-------|
| K-Means | 0.2510 | Chosen method |
| Hierarchical | 0.2485 | Similar performance |
| DBSCAN | Varies | Depends on eps parameter |

**Key Results:**
- ✅ **Optimal K:** 3 clusters identified via Silhouette Analysis
- ⚠️ **Silhouette Score:** 0.2510 indicates **poor clustering quality** - dataset has weak natural cluster structure
- ✅ Three distinct customer segments identified with clear behavioral differences
- ✅ PCA visualization shows cluster overlap, explaining low silhouette score

**Business Insights:**
1. **Cluster 0 (Premium):** Target for premium services, loyalty programs, high-value offers
2. **Cluster 1 (Regular):** Engagement campaigns, increase usage frequency, cross-sell opportunities  
3. **Cluster 2 (Cash Advance):** Risk monitoring, financial counseling, debt management programs

**Limitations:**
- Low silhouette score suggests customers don't naturally form tight, well-separated groups
- Overlapping behavior patterns between segments
- Feature engineering or additional data might improve segmentation quality

---

## 📊 Summary of Models and Metrics

### Classification (Fraud Detection)

**Problem Type:** Binary Classification with Severe Class Imbalance

**Models Evaluated:**
1. Random Forest Classifier (n_estimators=100, max_depth=10, class_weight='balanced')
2. XGBoost Classifier (n_estimators=150, max_depth=6, scale_pos_weight=3.32)

**Evaluation Metrics:**
- **ROC-AUC Score:** Measures model's ability to distinguish between classes (higher is better, max=1.0)
- **Precision:** Percentage of predicted frauds that are actually frauds (reduces false alarms)
- **Recall:** Percentage of actual frauds correctly detected (minimizes missed frauds)
- **F1-Score:** Harmonic mean of precision and recall (balanced metric)
- **Confusion Matrix:** Visualizes true positives, false positives, true negatives, false negatives

**Winner:** XGBoost (ROC-AUC: 0.9905, F1: 0.9530)

---

### Regression (Year Prediction)

**Problem Type:** Regression with Continuous Target Variable (Year: 1922-2011)

**Models Evaluated:**
1. Linear Regression (baseline model)
2. Random Forest Regressor (n_estimators=100, max_depth=15)
3. XGBoost Regressor (n_estimators=150, max_depth=8, learning_rate=0.1)

**Evaluation Metrics:**
- **RMSE (Root Mean Squared Error):** Average prediction error in years (lower is better)
- **MAE (Mean Absolute Error):** Average absolute prediction error in years (more interpretable)
- **R² Score:** Proportion of variance explained by model (0-1, higher is better)
- **Residual Plots:** Visualizes prediction errors and identifies systematic biases

**Winner:** XGBoost (R²: 0.3553, RMSE: 8.76 years, MAE: 6.16 years)

---

### Clustering (Customer Segmentation)

**Problem Type:** Unsupervised Learning - Customer Grouping

**Algorithms Evaluated:**
1. K-Means Clustering (n_clusters=3, n_init=10)
2. Hierarchical Clustering (Ward linkage)
3. DBSCAN (eps=3.0, min_samples=10)

**Evaluation Metrics:**
- **Silhouette Score:** Measures cluster cohesion and separation (-1 to 1, higher is better)
  - 0.71-1.0: Strong structure
  - 0.51-0.70: Reasonable structure
  - 0.26-0.50: Weak structure
  - <0.25: No substantial structure
- **Inertia:** Within-cluster sum of squares (lower indicates tighter clusters)
- **Elbow Method:** Visual technique to identify optimal K
- **Cluster Size Distribution:** Ensures balanced or meaningful segment sizes

**Winner:** K-Means with K=3 (Silhouette: 0.2510 - poor quality but best available)

---
