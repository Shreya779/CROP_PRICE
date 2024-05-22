import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load the dataset into a pandas DataFrame
df = pd.read_csv("D:/EDAI/maharashtra.csv")

print("Missing values in each column:")
print(df.isnull().sum())

# Impute missing values using KNNImputer
imputer = KNNImputer()
df['Production'] = imputer.fit_transform(df[['Production']])

print("\nMissing values after imputation:")
print(df.isnull().sum())

# One-hot encode categorical columns
categorical_columns = ['District_Name', 'Season', 'Crop']
df = pd.get_dummies(df, columns=categorical_columns)

# Separate features (X) and target variable (y)
X = df.drop('Production', axis=1)  # Assuming 'Production' is the target variable
y = df['Production']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor model with hyperparameter tuning and regularization
rf_regr = RandomForestRegressor(n_estimators=100,
                                 max_depth=10,
                                 min_samples_split=5,  # Adjust min_samples_split for regularization
                                 min_samples_leaf=2,   # Adjust min_samples_leaf for regularization
                                 max_features='sqrt',  # Use 'sqrt' for regularization
                                 random_state=42)

# Train the Random Forest model
rf_regr.fit(X_train, y_train)

# Make predictions using Random Forest
y_pred_train_rf = rf_regr.predict(X_train)
y_pred_test_rf = rf_regr.predict(X_test)

# Evaluate Random Forest model
print("Training R2 Score (Random Forest):", r2_score(y_train, y_pred_train_rf))
print("Test R2 Score (Random Forest):", r2_score(y_test, y_pred_test_rf))

# Calculate Mean Squared Error
mse_train = mean_squared_error(y_train, y_pred_train_rf)
mse_test = mean_squared_error(y_test, y_pred_test_rf)

print("Training Mean Squared Error (Random Forest):", mse_train)
print("Test Mean Squared Error (Random Forest):", mse_test)

# Cross-validation
cv_scores = cross_val_score(rf_regr, X, y, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()

print("Cross-Validation Mean Squared Error:", cv_mse)
