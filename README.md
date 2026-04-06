<h1 align="center">Credit Card Customer Profiling & Segmentation</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.7-orange?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Clustering-KMeans%20%7C%20Hierarchical-green" />
  <img src="https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" />
</p>

<p align="center">
  An end-to-end unsupervised Machine Learning project that segments <strong>8,950 credit card customers</strong> into distinct behavioural groups — enabling targeted marketing strategies and personalised financial products.
</p>

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Clustering Results](#clustering-results)
- [Customer Segment Profiles](#customer-segment-profiles)
- [Business Insights](#business-insights)
- [Setup & Usage](#setup--usage)
- [Tech Stack](#tech-stack)

---

## Problem Statement

A bank has **8,950 active credit card holders** with 18 behavioural features collected over 6 months. There are no predefined labels. The goal is to:

> **Discover natural customer segments** based on spending habits, payment behaviour, and credit usage — so the bank can target each group with tailored products and campaigns.

---

## Dataset

**Source:** [Kaggle — Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)  
**Records:** 8,950 customers | **Features:** 18 | **Period:** 6-month snapshot

| Feature | Type | Description |
|---|---|---|
| BALANCE | Monetary | Amount left for purchases |
| PURCHASES | Monetary | Total purchase amount |
| ONEOFF_PURCHASES | Monetary | Largest single purchase |
| INSTALLMENTS_PURCHASES | Monetary | Total installment purchases |
| CASH_ADVANCE | Monetary | Total cash advances taken |
| CREDIT_LIMIT | Monetary | Customer credit limit |
| PAYMENTS | Monetary | Total payments made |
| MINIMUM_PAYMENTS | Monetary | Minimum payments made |
| PURCHASES_FREQUENCY | Frequency | How often purchases are made (0–1) |
| CASH_ADVANCE_FREQUENCY | Frequency | How often cash advances are taken (0–1) |
| PRC_FULL_PAYMENT | Frequency | % of months with full payment made |
| TENURE | Count | Duration of credit card service (months) |

---

## Project Structure

```
Credit-Card-Customer-Profiling/
│
├── data/
│   ├── raw/                              # Original dataset (gitignored)
│   └── processed/                        # Cleaned & scaled data (gitignored)
│
├── notebooks/
│   ├── 01_Data_Preparation.ipynb         # Load, inspect, EDA
│   ├── 02_Feature_Engineering.ipynb      # Imputation, 7 ratio features, scaling
│   ├── 03_KMeans_Clustering.ipynb        # Elbow, Silhouette, KMeans K=4, PCA
│   ├── 04_Hierarchical_Clustering.ipynb  # Ward linkage + comparison with KMeans
│   └── 05_Customer_Profiling.ipynb       # Cluster profiles & business strategies
│
├── src/
│   ├── config.py                         # Paths, constants, parameters
│   ├── data_loader.py                    # Load & validate raw data
│   ├── preprocessing.py                  # Clean, impute, engineer, scale
│   ├── model.py                          # KMeans, Hierarchical, PCA
│   └── visualize.py                      # All plot functions (saves to images/)
│
├── images/                               # All generated plots (committed to GitHub)
├── models/                               # Saved models: kmeans.pkl, pca.pkl, scaler.pkl
├── reports/
│   └── cluster_summary.csv              # Mean feature values per segment
│
├── scripts/
│   └── download_data.py                 # Kaggle API download
│
├── main.py                              # Full pipeline runner
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Workflow

```
Raw Data
   │
   ▼
01_Data_Preparation      ── Missing value analysis, distributions, correlations
   │
   ▼
02_Feature_Engineering   ── Median imputation, 7 ratio features, StandardScaler
   │
   ▼
03_KMeans_Clustering     ── Elbow Method, Silhouette Score, K=4, PCA 2D view
   │
   ▼
04_Hierarchical_Clustering ── Ward linkage, side-by-side comparison
   │
   ▼
05_Customer_Profiling    ── Segment labels, heatmap, business strategies
   │
   ▼
main.py                  ── End-to-end pipeline (single command)
```

---

## Exploratory Data Analysis

### Missing Values
**MINIMUM_PAYMENTS** (3.5%) and **CREDIT_LIMIT** (0.01%) had missing values — imputed with column median.

![Missing Values](images/01_missing_values.png)

### Monetary Feature Distributions
All monetary features are **heavily right-skewed** — most customers have moderate values while a small group has very high values. This confirms the need for StandardScaler before clustering.

![Distributions](images/02_distributions.png)

### Frequency Features
Approximately **50% of customers never take cash advances** — a strong segmentation signal. Only ~22% of customers always pay in full.

![Frequency Features](images/03_frequency_features.png)

### Correlation Heatmap

![Correlation Heatmap](images/04_correlation_heatmap.png)

---

## Feature Engineering

7 new behavioural ratio features were created from domain knowledge:

| Feature | Formula | Business Meaning |
|---|---|---|
| `PURCHASES_TO_LIMIT_RATIO` | PURCHASES / CREDIT_LIMIT | Credit utilisation for spending |
| `CASH_ADVANCE_RATIO` | CASH_ADVANCE / BALANCE | Cash reliance vs balance held |
| `PAYMENT_TO_MINIMUM_RATIO` | PAYMENTS / MIN_PAYMENTS | Debt repayment aggressiveness |
| `MONTHLY_AVG_PURCHASE` | PURCHASES / TENURE | Average monthly spending rate |
| `INSTALLMENT_TO_PURCHASE_RATIO` | INSTALLMENTS / PURCHASES | Preference for installment buying |
| `CASH_ADVANCE_TO_CREDIT_RATIO` | CASH_ADVANCE / CREDIT_LIMIT | Cash usage vs credit limit |
| `BALANCE_TO_CREDIT_RATIO` | BALANCE / CREDIT_LIMIT | Balance utilisation rate |

![Engineered Features](images/05_engineered_features.png)

### Outlier Analysis
Outliers were **retained** — they represent genuine high-value or high-risk customers, not data errors.

![Outliers](images/06_outliers.png)

---

## Clustering Results

### Finding Optimal K
Both Elbow Method and Silhouette Score were used to determine K=4 — balancing statistical fit with business interpretability.

![Optimal K](images/07_optimal_k.png)

### PCA Variance
10 principal components explain **81% of total variance**.

![PCA Variance](images/11_pca_variance.png)

### KMeans Segments in 2D (PCA)

![KMeans PCA](images/08_clusters_pca_kmeans.png)

### KMeans vs Hierarchical Clustering

| Algorithm | Silhouette Score | Notes |
|---|---|---|
| **KMeans** | **0.1758** | Better cluster separation, scalable |
| Hierarchical (Ward) | 0.1050 | Confirms segment structure, deterministic |

![Hierarchical PCA](images/09_clusters_pca_hierarchical.png)

---

## Customer Segment Profiles

### Segment Heatmap & Key Metrics

![Cluster Profiles](images/09_cluster_profiles.png)

### Segment Size Distribution

![Cluster Distribution](images/10_cluster_distribution.png)

---

## Business Insights

| Cluster | Segment | Size | Avg Balance | Avg Purchases | Avg Cash Advance | Strategy |
|---|---|---|---|---|---|---|
| 0 | **Low Activity** | 3,917 (43.8%) | $984 | $248 | $573 | Re-engagement offers, spend incentives |
| 1 | **Active Transactors** | 3,475 (38.8%) | $948 | $1,376 | $216 | Premium rewards, travel miles, limit upgrades |
| 2 | **VIP / High Spenders** | 278 (3.1%) | $3,820 | $9,389 | $773 | Concierge VIP card, relationship manager |
| 3 | **Cash Advance Heavy** | 1,280 (14.3%) | $4,523 | $482 | $4,338 | Debt consolidation, financial counselling |

### Key Takeaways

1. **VIP High Spenders** (3%) generate disproportionate revenue — retention is the top priority
2. **Cash Advance Heavy** (14%) carry the highest financial risk — proactive outreach reduces default
3. **Low Activity** (44%) is the largest segment — small activation campaigns have outsized impact
4. **Active Transactors** (39%) are ideal premium card candidates — high spend, regular payers
5. StandardScaler was **essential** — without scaling, high-value monetary features dominated distance calculations and produced meaningless clusters

---

## Setup & Usage

### 1. Clone & Install
```bash
git clone https://github.com/bhavesh2418/Credit-Card-Customer-Profiling.git
cd Credit-Card-Customer-Profiling
pip install -r requirements.txt
```

### 2. Configure Credentials
```bash
cp .env.example .env
# Fill in your Kaggle API credentials and GitHub token
```

### 3. Download Dataset
```bash
python scripts/download_data.py
```

### 4. Run Full Pipeline
```bash
python main.py
```
This generates all 11 plots in `images/` and saves `reports/cluster_summary.csv`.

### 5. Explore Notebooks
Open notebooks in order: `01` → `02` → `03` → `04` → `05`

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas, NumPy | Data manipulation |
| Scikit-learn | KMeans, Hierarchical, PCA, StandardScaler, Silhouette |
| Matplotlib, Seaborn | All visualisations |
| Joblib | Model persistence |
| Jupyter Notebook | Interactive analysis |
| Kaggle API | Dataset download |

---

<p align="center">
  <strong>Bhavesh</strong> — Portfolio project demonstrating unsupervised ML, customer segmentation, and production-style workflow.<br>
  <a href="https://github.com/bhavesh2418">GitHub</a>
</p>
