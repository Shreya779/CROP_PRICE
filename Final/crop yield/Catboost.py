import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor  # Import CatBoost

# Load the dataset into a pandas DataFrame
df1 = pd.read_csv("D:/EDAI/processed_apy_maharashtra.csv")

# Apply Random Forest imputation to fill missing values in 'Production' column
rf_regressor = RandomForestRegressor()
imputer = IterativeImputer(estimator=rf_regressor)
df1['Production'] = imputer.fit_transform(df1[['Production']])

# Convert categorical features to the 'category' dtype
df1['District_Name'] = df1['District_Name'].astype('category')
df1['Season'] = df1['Season'].astype('category')
df1['Crop'] = df1['Crop'].astype('category')

# Encode categorical variables
le_district = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

df1['District_Name'] = le_district.fit_transform(df1['District_Name'])
df1['Season'] = le_season.fit_transform(df1['Season'])
df1['Crop'] = le_crop.fit_transform(df1['Crop'])

# Split the data into training and testing sets
X = df1[['Crop_Year', 'District_Name', 'Season', 'Crop', 'Area']]
y = df1['Production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost model for regression with increased complexity (intentional overfitting)
cat_regr_overfit = CatBoostRegressor(iterations=1000, depth=10, learning_rate=0.01, random_state=42)

# Train the overfit CatBoost model
cat_regr_overfit.fit(X_train, y_train, cat_features=['District_Name', 'Season', 'Crop'])

# Make predictions using CatBoost
y_pred_train_cat_overfit = cat_regr_overfit.predict(X_train)
y_pred_test_cat_overfit = cat_regr_overfit.predict(X_test)

# Evaluate overfit CatBoost model
print("Training Accuracy (Overfit CatBoost):", r2_score(y_train, y_pred_train_cat_overfit))
print("Test Accuracy (Overfit CatBoost):", r2_score(y_test, y_pred_test_cat_overfit))
