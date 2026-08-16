# 🏠 California House Price Prediction

An end-to-end machine learning project that predicts median house value for California census block groups, using the classic California Housing dataset. The project covers data cleaning, EDA, feature engineering, model comparison, hyperparameter tuning, and deployment as a live web app.

**🔗 Live App:** [https://california-house-price-prediction-ag.streamlit.app/](https://california-house-price-prediction-ag.streamlit.app/)
**💻 Notebooks:** see `/notebooks`

---

## Problem Statement

Given census-level data for a California block group (location, housing age, room/bedroom counts, population, income, proximity to the ocean), predict the median house value. Framed as a **regression** problem.

## Dataset

- 20,640 records, 10 columns: `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`, `ocean_proximity`, `median_house_value` (target)
- No duplicate rows
- `total_bedrooms` had 207 missing values (~1%)

## Project Structure

```
├── app.py                      # Streamlit web app
├── train_model.py              # Train-or-predict pipeline script
├── model.pkl                   # Trained XGBoost model
├── pipeline.pkl                # Preprocessing pipeline
├── requirements.txt
├── NoteBooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   └── 04_Model_Training.ipynb
└── data/
    ├── collected_data/
    │   └── housing.csv                  # Raw source dataset
    │
    ├── cleaned/
    │   └── cleaned_data.csv             # Output of 01_Data_Cleaning
    │
    ├── training_data/
    │   └── train_data.csv               # Full training portion (pre feature-engineering split)
    │
    ├── processed/
    │   ├── x_train_processed.csv        # Feature-engineered train features
    │   ├── y_train.csv
    │   ├── x_test_processed.csv         # Feature-engineered test features
    │   └── y_test.csv
    │
    ├── deployment_holdout/
    │   ├── deployment_test_data.csv     # Held out before any feature engineering
    │   └── deployment_test_labels.csv
    │
    └── input_output_csv/
        ├── input.csv                    # Sample input for train_model.py prediction mode
        ├── input_labels.csv
        └── output_data.csv              # Sample predictions
```

## Approach

### 1. Data Cleaning
- Loaded the raw dataset (20,640 rows, 10 columns), checked structure, dtypes, and summary statistics.
- Found 207 missing values in `total_bedrooms` (~1%) — flagged for median imputation during feature engineering rather than dropping rows.
- No duplicate records.
- `ocean_proximity` has 5 categories; `ISLAND` is a severely rare category with only 5 records — kept.

### 2. Exploratory Data Analysis
- **Target (`median_house_value`) is right-skewed with a hard cap at $500,000** — a visible spike at the maximum value confirms the target was censored during data collection, not a natural distribution peak. This is a real limitation of the source data: the model cannot learn to distinguish a $520K house from a $2M house, since both are recorded identically in training data.
- `housing_median_age` shows the same censoring pattern, capped at 52 years.
- All four raw count features (`total_rooms`, `total_bedrooms`, `population`, `households`) are highly right-skewed (skew 3.4–4.9) with visible high-value outliers, and are strongly correlated with **each other** — a multicollinearity signal, since they largely reflect block-group *size* rather than distinct information.
- `median_income` is the single strongest predictor of price (correlation ≈ 0.69) — consistent with intuition and every published analysis of this dataset.
- `longitude` and `latitude` show a strong negative correlation (-0.92) with each other, and geographic plots show clear coastal-vs-inland price clustering that isn't fully captured by their individual (weak) correlation with price.
- `ISLAND` properties have the highest median price despite only 5 records; `INLAND` has the lowest.

### 3. Feature Engineering
- Carved out a **deployment holdout (10%)** using `StratifiedShuffleSplit` on an income bracket, *before* any preprocessing — kept untouched to simulate genuinely unseen future data.
- Split the remainder into train/test.
- Imputed `total_bedrooms` missing values with the **median**, fit on train only.
- One-hot encoded `ocean_proximity`.
- Applied **Yeo-Johnson power transformation** to the 5 heavily skewed features (`total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`), fit on train and reused (not refit) on the test set to avoid leakage.
- All transformers combined into a single `ColumnTransformer`/`Pipeline` object for consistent, leakage-safe application at both training and prediction time.

### 4. Model Training & Comparison
Compared 5 regression models using 10-fold cross-validation (RMSE):

| Model              | RMSE       |
|---------------------|-----------|
| **XGBoost**          | **47,691.88** |
| Random Forest         | 48,958.60 |
| Decision Tree          | 67,636.03 |
| Ridge Regression       | 70,573.59 |
| Linear Regression       | 70,575.25 |

XGBoost was the clear winner, consistent with tree-boosting methods' general strength on structured/tabular data with non-linear relationships (e.g. the geographic and income effects seen in EDA).

### 5. Hyperparameter Tuning
Used `GridSearchCV` (10-fold CV, 243 parameter combinations, 2,430 total fits) over `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`:

- **Best parameters:** `n_estimators=700, max_depth=7, learning_rate=0.05, subsample=1.0, colsample_bytree=0.8`
- **CV RMSE improved:** 47,691.88 → 45,141.46

### 6. Final Results

| Metric | Value |
|---|---|
| **Final Test RMSE** | **≈ 46,670** |

The final model was evaluated once on a held-out test set that was never used during model comparison or tuning. As with the target's $500,000 cap noted in EDA, prediction error is expected to be higher on the highest end of the market, since the training data itself cannot represent true values above the cap.

## Deployment

The final XGBoost model and preprocessing pipeline are saved with `joblib` and served through a **Streamlit** app, where a user inputs a block group's location, demographics, and housing details to get an instant predicted median house value.

```bash
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model from scratch:
```bash
python train_model.py
```

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `matplotlib` / `seaborn` · `Streamlit` · `joblib`

## Known Limitations

- The target variable is capped at $500,000 in the source data — the model cannot accurately predict values above this ceiling, since it was never shown genuine examples above it during training.
- `ISLAND` properties (5 records) are too few to draw reliable conclusions from; predictions for this category should be treated with caution.

## Author

Abhishek Gawate
