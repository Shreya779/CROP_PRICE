import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

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
