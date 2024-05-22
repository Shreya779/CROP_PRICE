import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# Load the dataset into a pandas DataFrame
df = pd.read_csv("D:/EDAI/maharashtra.csv")

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

# XGBoost Regressor model with hyperparameter tuning, regularization, and early stopping
xgb_regr = XGBRegressor(
    n_estimators=1000,  # Adjust the number of boosting rounds
    learning_rate=0.1,  # Adjust the learning rate
    max_depth=5,        # Adjust the maximum depth of trees
    reg_alpha=1,        # Add L1 regularization
    reg_lambda=1,       # Add L2 regularization
    early_stopping_rounds=10,  # Early stopping to prevent overfitting
    eval_metric="rmse",        # Evaluation metric
    eval_set=[(X_test, y_test)],  # Validation set for early stopping
    verbose=True
)

# Train the XGBoost model
xgb_regr.fit(X_train, y_train)

# Make predictions using XGBoost
y_pred_train_xgb = xgb_regr.predict(X_train)
y_pred_test_xgb = xgb_regr.predict(X_test)

# Evaluate XGBoost model
print("Training Accuracy (XGBoost):", r2_score(y_train, y_pred_train_xgb))
print("Test Accuracy (XGBoost):", r2_score(y_test, y_pred_test_xgb))
