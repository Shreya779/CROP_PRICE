import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.utils import resample

# Load the dataset into a pandas DataFrame
df = pd.read_csv("D:/EDAI/processed_maharashtra.csv")

# Impute missing values using Random Forest Regressor
rf_regressor = RandomForestRegressor()
imputer = IterativeImputer(estimator=rf_regressor)
df['Production'] = imputer.fit_transform(df[['Production']])

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['District_Name', 'Season', 'Crop'])

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Separate features (X) and target variable (y)
X = df.drop('Production', axis=1)  # Assuming 'Production' is the target variable
y = df['Production']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost Regressor model
xgb_regr = XGBRegressor()

# Train the XGBoost model
xgb_regr.fit(X_train, y_train)

# Make predictions using XGBoost
y_pred_train_xgb = xgb_regr.predict(X_train)
y_pred_test_xgb = xgb_regr.predict(X_test)

# Evaluate XGBoost model
print("Training Accuracy (XGBoost):", r2_score(y_train, y_pred_train_xgb))
print("Test Accuracy (XGBoost):", r2_score(y_test, y_pred_test_xgb))

# Set the number of bootstrap samples
num_bootstrap_samples = 10

# Initialize lists to store in-sample and out-of-sample scores
in_sample_scores = []
out_of_sample_scores = []

# Perform bootstrapping
for i in range(num_bootstrap_samples):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_train, y_train, random_state=i)

    # Train an XGBoost model on the bootstrap sample
    xgb_regr_bootstrap = XGBRegressor()
    xgb_regr_bootstrap.fit(X_bootstrap, y_bootstrap)

    # Evaluate the model on the in-sample data
    y_pred_train_bootstrap = xgb_regr_bootstrap.predict(X_bootstrap)
    in_sample_scores.append(r2_score(y_bootstrap, y_pred_train_bootstrap))

    # Evaluate the model on the out-of-sample data
    y_pred_test_bootstrap = xgb_regr_bootstrap.predict(X_test)
    out_of_sample_scores.append(r2_score(y_test, y_pred_test_bootstrap))

# Calculate and print the average in-sample and out-of-sample scores
average_in_sample_score = np.mean(in_sample_scores)
average_out_of_sample_score = np.mean(out_of_sample_scores)

print("Average In-Sample Score:", average_in_sample_score)
print("Average Out-of-Sample Score:", average_out_of_sample_score)
