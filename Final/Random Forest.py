import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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

# Random Forest Regressor model
rf_regr = RandomForestRegressor()

# Train the Random Forest model
rf_regr.fit(X_train, y_train)

# Make predictions using Random Forest
y_pred_train_rf = rf_regr.predict(X_train)
y_pred_test_rf = rf_regr.predict(X_test)

# Evaluate Random Forest model
print("Training Accuracy (Random Forest):", r2_score(y_train, y_pred_train_rf))
print("Test Accuracy (Random Forest):", r2_score(y_test, y_pred_test_rf))
