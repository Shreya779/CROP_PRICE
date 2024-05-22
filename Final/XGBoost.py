import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset into a pandas DataFrame
df1 = pd.read_csv("D:/EDAI/processed_apy_maharashtra.csv")

X = df1[['Crop_Year', 'District_Name', 'Season', 'Crop', 'Area']]
y = df1['Production']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model for regression
xgb_regr = XGBRegressor()

# Train the model
xgb_regr.fit(X_train, y_train)

# Make predictions
y_pred_train_xgb = xgb_regr.predict(X_train)
y_pred_test_xgb = xgb_regr.predict(X_test)

# Evaluate the model
print("Training Accuracy (XGBoost):", r2_score(y_train, y_pred_train_xgb))
print("Test Accuracy (XGBoost):", r2_score(y_test, y_pred_test_xgb))









# import pandas as pd
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# from sklearn.metrics import r2_score
# from sklearn.preprocessing import LabelEncoder
#
# # Load the imputed DataFrame
# df_imputed = pd.read_csv("D:/EDAI/imputed_apy.csv")
#
# # Convert categorical features to the 'category' dtype
# df_imputed['District_Name'] = df_imputed['District_Name'].astype('category')
# df_imputed['Season'] = df_imputed['Season'].astype('category')
# df_imputed['Crop'] = df_imputed['Crop'].astype('category')
#
# # Separate features (X) and target variable (y)
# X = df_imputed[['Crop_Year', 'District_Name', 'Season', 'Crop', 'Area']]
# y = df_imputed['Production']
#
# # Divide the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # XGBoost model for regression with categorical support enabled
# xgb_regr = XGBRegressor(enable_categorical=True)
#
# # Train the model
# xgb_regr.fit(X_train, y_train)
#
# # Make predictions
# y_pred_train_xgb = xgb_regr.predict(X_train)
# y_pred_test_xgb = xgb_regr.predict(X_test)
#
# # Evaluate the model
# print("Training Accuracy (XGBoost):", r2_score(y_train, y_pred_train_xgb))
# print("Test Accuracy (XGBoost):", r2_score(y_test, y_pred_test_xgb))

