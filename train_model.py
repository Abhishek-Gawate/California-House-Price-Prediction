import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from xgboost import XGBRegressor


MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline():

    # Columns that need Yeo-Johnson transformation
    power_columns = ["total_rooms",
                     "total_bedrooms",
                     "population",
                     "households",
                     "median_income"]

    # Numerical columns that don't need transformation
    normal_numeric_columns = ["longitude",
                              "latitude",
                              "housing_median_age"]

    # Categorical column
    categorical_columns = ["ocean_proximity"]


    # Yeo-Johnson pipeline
    power_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("power_transformer",PowerTransformer(method="yeo-johnson",standardize=False))])


    # Normal numerical pipeline
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])


    # Categorical pipeline
    categorical_pipeline = Pipeline([("one_hot",OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore"))])


    # Combine everything
    full_pipeline = ColumnTransformer([("power", power_pipeline, power_columns),
                                       ("numeric",numeric_pipeline,normal_numeric_columns),
                                       ("categorical",categorical_pipeline,categorical_columns)])
    return full_pipeline


# TRAINING MODEL

if not os.path.exists(MODEL_FILE):
    # Loading training dataset
    housing = pd.read_csv("data/training_data/train_data.csv")

    # Create income category for stratified split
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])

    # Create 10% test dataset
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.10,random_state=42)

    for train_index, test_index in split.split(housing,housing["income_cat"]):
        
        x_train = housing.loc[train_index].reset_index(drop=True)
        x_test = housing.loc[test_index].reset_index(drop=True)
		
    # Saved test dataset
    y_test = x_test["median_house_value"].copy()

    x_test = x_test.drop(columns=["median_house_value","income_cat"])

    y_test.to_csv("data/input_output_csv/input_labels.csv",index=False)

    x_test.to_csv("data/input_output_csv/input.csv",index=False)

    # Training data
    housing_labels = x_train["median_house_value"].copy()

    housing_features = x_train.drop(columns=["median_house_value","income_cat"])

    # Build preprocessing pipeline
    pipeline = build_pipeline()
	
    housing_prepared = pipeline.fit_transform(housing_features)
	
    # Final tuned XGBoost model
    model = XGBRegressor(n_estimators=700,
                         max_depth=7,
                         learning_rate=0.05,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         random_state=42)

    model.fit(housing_prepared,housing_labels)
	
	# Save model and pipeline
    joblib.dump(model,MODEL_FILE)

    joblib.dump(pipeline,PIPELINE_FILE)

    print("Model and pipeline saved successfully.")

# PREDICTION
else:
    # Load input data
    input_data = pd.read_csv("data/input_output_csv/input.csv")

    # Load saved model
    model = joblib.load(MODEL_FILE)
    # Load saved preprocessing pipeline
    pipeline = joblib.load(PIPELINE_FILE)

    # Apply transformations
    input_transformed = pipeline.transform(input_data)

    # Make predictions
    predictions = model.predict(input_transformed)

    # Add predictions
    input_data["house_prices"] = predictions

    # Save output
    input_data.to_csv("data/input_output_csv/output_data.csv",index=False)

    print("Predictions saved to output_data.csv")