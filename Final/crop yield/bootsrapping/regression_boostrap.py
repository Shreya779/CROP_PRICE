import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.metrics import r2_score
import numpy as np

# Load the dataset into a pandas DataFrame
df = pd.read_csv("D:/EDAI/maharashtra.csv")

# Impute missing values using IterativeImputer
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

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Polynomial Regression
degree = 2  # Adjust the degree as needed
poly = PolynomialFeatures(degree)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Apply Ridge Regression with cross-validation
ridge_regr = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge_model = make_pipeline(PolynomialFeatures(degree), ridge_regr)
ridge_model.fit(X_train_scaled, y_train)

# Make predictions using Ridge Regression
y_pred_train_ridge = ridge_model.predict(X_train_scaled)
y_pred_test_ridge = ridge_model.predict(X_test_scaled)

# Evaluate Ridge Regression model
print("Training Accuracy (Ridge Regression):", r2_score(y_train, y_pred_train_ridge))
print("Test Accuracy (Ridge Regression):", r2_score(y_test, y_pred_test_ridge))

# Set the number of bootstrap samples
num_bootstrap_samples = 10

# Initialize lists to store in-sample and out-of-sample scores
in_sample_scores = []
out_of_sample_scores = []

# Perform bootstrapping
for i in range(num_bootstrap_samples):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_train_scaled, y_train, random_state=i)

    # Train a Ridge Regression model on the bootstrap sample
    ridge_regr_bootstrap = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    ridge_model_bootstrap = make_pipeline(PolynomialFeatures(degree), ridge_regr_bootstrap)
    ridge_model_bootstrap.fit(X_bootstrap, y_bootstrap)

    # Evaluate the model on the in-sample data
    y_pred_train_bootstrap = ridge_model_bootstrap.predict(X_bootstrap)
    in_sample_scores.append(r2_score(y_bootstrap, y_pred_train_bootstrap))

    # Evaluate the model on the out-of-sample data
    y_pred_test_bootstrap = ridge_model_bootstrap.predict(X_test_scaled)
    out_of_sample_scores.append(r2_score(y_test, y_pred_test_bootstrap))

# Calculate and print the average in-sample and out-of-sample scores
average_in_sample_score = np.mean(in_sample_scores)
average_out_of_sample_score = np.mean(out_of_sample_scores)

print("Average In-Sample Score:", average_in_sample_score)
print("Average Out-of-Sample Score:", average_out_of_sample_score)
