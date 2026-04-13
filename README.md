# Amazon Customer Purchase Behaviour Analysis & Prediction

A complete end-to-end machine learning pipeline for analysing Amazon customer data — covering data cleaning, feature engineering, customer segmentation, CLV prediction, and churn classification. Built in Python using scikit-learn, with all trained models exported as `.pkl` files ready for deployment in a Streamlit or Flask application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Model Results](#model-results)
- [Feature Engineering](#feature-engineering)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Output Files](#output-files)
- [Key Design Decisions](#key-design-decisions)

---

## Project Overview

This project analyses purchase behaviour of Amazon customers and builds three predictive models:

| Model | Type | Goal |
|-------|------|------|
| CLV Prediction | Linear Regression | Predict Customer Lifetime Value |
| Churn Prediction | Random Forest + SMOTE | Predict whether a customer will churn |
| Customer Segmentation | KMeans + PCA | Group customers into 3 meaningful segments |

All models are saved as `.pkl` files and scalers are saved alongside them, making them directly deployable in a web application.

---

## Dataset

**File:** `amazon.xlsx`

| Property | Value |
|----------|-------|
| Rows | 1,591 |
| Columns | 19 |
| Churn distribution | 797 churned / 794 not churned (balanced) |
| CLV range | 56 – 7,204 |
| Loyalty Score range | 1 – 100 |
| Missing values | Customer_Name (77), Feedback_Comments (709) |

### Columns

```
Customer_ID, Customer_Name, Age, Gender, Location, Product_Category,
Product_ID, Purchase_Date, Purchase_Amount, Payment_Method, Rating,
Feedback_Comments, Customer_Lifetime_Value, Loyalty_Score,
Discount_Applied, Return_Status, Customer_Segment,
Preferred_Shopping_Channel, Churn
```

---

## Project Structure

```
amazonnew/
├── amazonnew.ipynb          # Main analysis notebook (32 cells)
├── amazon.xlsx              # Raw input dataset (never overwritten)
├── amazonnew_output.xlsx    # Final engineered DataFrame (Excel)
├── amazon_cleaned.csv       # Final engineered DataFrame (CSV)
├── clv_model.pkl            # Trained Linear Regression model
├── clv_scaler.pkl           # StandardScaler for CLV features
├── churn_model.pkl          # Trained Random Forest classifier
├── churn_scaler.pkl         # StandardScaler for churn features
├── kmeans_model.pkl         # Trained KMeans clustering model
├── kmeans_scaler.pkl        # StandardScaler for clustering features
└── README.md
```

---

## Pipeline Walkthrough

### 1. Data Loading & Cleaning

- Loads `amazon.xlsx` and removes completely empty rows
- Strips whitespace from all column names
- Fills missing numeric values (`Age`, `Rating`, `Purchase_Amount`) with **median**
- Fills missing categorical values (`Payment_Method`) with **mode**
- Removes duplicate records on `Customer_ID` + `Purchase_Date`
- Standardises `Gender` capitalisation
- Label-encodes categorical columns: `Gender`, `Payment_Method`, `Preferred_Shopping_Channel`
- Converts `Return_Status` to binary (Yes → 1, No → 0)
- Removes outliers using **Z-score < 3** on `Purchase_Amount`

### 2. Feature Engineering

Four new features are computed per customer and merged back into the main DataFrame:

| Feature | Calculation |
|---------|-------------|
| `CLV` | `Avg_Purchase × Order_Count` (engineered proxy) |
| `Purchase_Frequency` | Count of transactions per customer |
| `Avg_Purchase_Value` | Mean purchase amount per customer |
| `Days_Since_Purchase` | Days from last purchase to today |

The raw `Customer_Lifetime_Value` column from the dataset is preserved and used as the **true CLV target** in model training.

### 3. Customer Segmentation (KMeans + PCA)

- Features: `Customer_Lifetime_Value`, `Loyalty_Score`, `Purchase_Amount`
- Scaled with `StandardScaler`
- KMeans with `k=3` clusters (silhouette score: **0.348** — best of all tested configurations)
- PCA reduces to 2 components for visualisation
- Segment labels written back to main DataFrame

### 4. CLV Prediction (Linear Regression)

- Target: `Customer_Lifetime_Value` (real column, range 56–7,204)
- All remaining string columns encoded before scaling
- 80/20 train-test split, `random_state=42`

**Features used (9):**
```
Age, Purchase_Amount, Payment_Method, Loyalty_Score,
Rating, Gender, Return_Status, Customer_Segment,
Days_Since_Purchase
```

> Note: `Discount_Applied` is excluded — it has a constant value of 0.1 across all rows (zero variance, zero predictive power).

### 5. Churn Prediction (Random Forest + SMOTE)

- Target: real `Churn` column from dataset (binary 0/1)
- **SMOTE** applied to training set to balance classes (637 vs 637 after resampling)
- 80/20 stratified split, `random_state=42`

**Features used (11):**
```
Age, Purchase_Amount, Payment_Method, Loyalty_Score,
Avg_Purchase_Value, Days_Since_Purchase, Rating,
Location, Product_Category, Return_Status, Customer_Segment
```

---

## Model Results

### CLV Prediction

| Metric | Value |
|--------|-------|
| R² Score | **0.7648** |
| MSE | 644,390.86 |
| MAE | 594.60 |
| MAPE | **27.80%** |
| Actual CLV range (test set) | 73.4 – 6,843.4 |
| Predicted CLV range (test set) | 91.1 – 5,030.7 |

### Churn Prediction

| Metric | Value |
|--------|-------|
| Accuracy | **85.58%** |
| Precision (Churn class) | 0.86 |
| Recall (Churn class) | 0.85 |
| F1-Score (Churn class) | **0.86** |
| Test set size | 319 (159 no-churn / 160 churn) |

**Full classification report:**

```
              precision    recall  f1-score   support

    No Churn       0.85      0.86      0.86       159
       Churn       0.86      0.85      0.86       160

    accuracy                           0.86       319
   macro avg       0.86      0.86      0.86       319
weighted avg       0.86      0.86      0.86       319
```

### Customer Segmentation

| Config | Features | Silhouette Score |
|--------|----------|-----------------|
| Current (used) | CLV, Loyalty_Score, Purchase_Amount | **0.348** |
| Expanded (tested) | + Rating, Order_Count | 0.215 |

The 3-feature configuration gives the best-separated clusters. Adding more features dilutes the signal and worsens cluster quality.

---

## Feature Engineering

### Why these CLV features?

Feature importance from a Random Forest trained on 15 candidate features:

| Feature | Importance |
|---------|-----------|
| Purchase_Amount | 76.8% |
| Days_Since_Purchase | 3.9% |
| Loyalty_Score | 3.7% |
| Age | 3.6% |
| Avg_Purchase_Value | 3.4% |
| Others | 8.6% |
| Discount_Applied | 0.0% |

`Purchase_Amount` dominates. `Days_Since_Purchase` is the most valuable engineered feature.

### Why these Churn features?

| Feature | Importance |
|---------|-----------|
| Purchase_Amount | 36.1% |
| CLV_Engineered | 19.7% |
| Avg_Purchase_Value | 18.5% |
| Loyalty_Score | 4.5% |
| Days_Since_Purchase | 4.4% |
| Age | 4.0% |
| Others | 12.8% |
| Discount_Applied | 0.0% |

`Avg_Purchase_Value` and `Days_Since_Purchase` together contribute 22.9% — this is why the expanded feature set beats the basic set by 1.5% accuracy.

---

## Technologies Used

```
Python 3.13
pandas
numpy
scikit-learn
imbalanced-learn (SMOTE)
matplotlib
seaborn
plotly
scipy
joblib
openpyxl
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn plotly scipy joblib openpyxl
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/amazon-customer-analysis.git
cd amazon-customer-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Place the dataset

Put `amazon.xlsx` in the root directory alongside the notebook.

### 4. Run the notebook

```bash
jupyter notebook amazonnew.ipynb
```

Run all cells top to bottom. The notebook will:
- Clean and engineer the data
- Train all three models
- Save 6 `.pkl` files
- Export `amazonnew_output.xlsx` and `amazon_cleaned.csv`

### 5. Use the saved models in your app

```python
import joblib
import numpy as np

# Load CLV model
clv_model  = joblib.load('clv_model.pkl')
clv_scaler = joblib.load('clv_scaler.pkl')

# Predict CLV for a new customer
# Features: Age, Purchase_Amount, Payment_Method, Loyalty_Score,
#           Rating, Gender, Return_Status, Customer_Segment, Days_Since_Purchase
sample = np.array([[35, 250.0, 2, 65, 4, 0, 0, 1, 180]])
clv_pred = clv_model.predict(clv_scaler.transform(sample))
print(f"Predicted CLV: {clv_pred[0]:.2f}")

# Load Churn model
churn_model  = joblib.load('churn_model.pkl')
churn_scaler = joblib.load('churn_scaler.pkl')

# Predict churn
# Features: Age, Purchase_Amount, Payment_Method, Loyalty_Score,
#           Avg_Purchase_Value, Days_Since_Purchase, Rating,
#           Location, Product_Category, Return_Status, Customer_Segment
sample_c = np.array([[35, 250.0, 2, 65, 210.0, 180, 4, 1, 2, 0, 1]])
churn_pred = churn_model.predict(churn_scaler.transform(sample_c))
print(f"Churn prediction: {'Will Churn' if churn_pred[0] == 1 else 'No Churn'}")

# Load KMeans model
kmeans_model  = joblib.load('kmeans_model.pkl')
kmeans_scaler = joblib.load('kmeans_scaler.pkl')

# Predict segment
# Features: Customer_Lifetime_Value, Loyalty_Score, Purchase_Amount
sample_k = np.array([[1200.0, 65, 250.0]])
segment = kmeans_model.predict(kmeans_scaler.transform(sample_k))
print(f"Customer segment: {segment[0]}")
```

---

## Output Files

| File | Description |
|------|-------------|
| `amazonnew_output.xlsx` | Full engineered DataFrame with segment labels (1,591 rows × 24 cols) |
| `amazon_cleaned.csv` | Same data in CSV format for Power BI / Tableau |
| `clv_model.pkl` | LinearRegression — CLV prediction |
| `clv_scaler.pkl` | StandardScaler fitted on CLV training features |
| `churn_model.pkl` | RandomForestClassifier — churn prediction |
| `churn_scaler.pkl` | StandardScaler fitted on churn training features |
| `kmeans_model.pkl` | KMeans (k=3) — customer segmentation |
| `kmeans_scaler.pkl` | StandardScaler fitted on segmentation features |

### Final DataFrame columns (24)

```
Customer_ID, Customer_Name, Age, Gender, Location, Product_Category,
Product_ID, Purchase_Date, Purchase_Amount, Payment_Method, Rating,
Feedback_Comments, Customer_Lifetime_Value, Loyalty_Score, Discount_Applied,
Return_Status, Customer_Segment, Preferred_Shopping_Channel, Churn,
CLV, Purchase_Frequency, Avg_Purchase_Value, Days_Since_Purchase, Segment
```

---

## Key Design Decisions

**Real churn labels, not synthetic ones.**
The dataset already contains a ground-truth `Churn` column (50.09% churn rate, nearly perfectly balanced). The model trains on real observed behaviour, not on a rule invented from CLV thresholds. This gives an honest 85.58% accuracy that reflects genuine predictive power.

**SMOTE on training data only.**
SMOTE is applied after the train-test split, exclusively on the training set. The test set contains the original real distribution. This is the correct methodology — applying SMOTE before splitting causes data leakage and inflates reported accuracy.

**`Customer_Lifetime_Value` as CLV target, not engineered proxy.**
The dataset provides a real CLV column (range 56–7,204). Using this as the prediction target gives MAPE of 27.8%. Using the engineered `CLV` proxy (range 10–1,217) gives MAPE of 73.4% — a 45-point difference caused entirely by the wrong target column.

**`Discount_Applied` excluded from all models.**
This column has a constant value of 0.1 for every single row in the dataset (standard deviation ≈ 0). It contributes 0.0% feature importance to both models and adding it to `StandardScaler` can cause numerical instability. It is intentionally excluded.

**Clustering features kept at 3.**
Testing showed that adding more features to KMeans drops the silhouette score from 0.348 → 0.215. `Customer_Lifetime_Value`, `Loyalty_Score`, and `Purchase_Amount` produce the most clearly separated customer segments.

---

## Acknowledgements

Dataset used for academic and learning purposes. Analysis conducted as part of a customer behaviour study using Amazon purchase data.
