import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset into a pandas DataFrame
df = pd.read_csv("D:/EDAI/maharashtra.csv")

# Print column names to verify
print("Column Names:", df.columns)

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

# Take input from the user for prediction
user_input_crop_year = int(input("Enter Crop Year: "))
user_input_district = input("Enter District: ")
user_input_season = input("Enter Season: ")
user_input_crop = input("Enter Crop: ")
user_input_area = float(input("Enter Area:"))

# Encode categorical variables for user input
le_district = LabelEncoder()
le_season = LabelEncoder()
le_crop = LabelEncoder()

# Fit the LabelEncoders with the corresponding columns
le_district.fit(df['District_Name'])
le_season.fit(df['Season'])
le_crop.fit(df['Crop'])

user_input_district_encoded = le_district.transform([user_input_district])[0]
user_input_season_encoded = le_season.transform([user_input_season])[0]
user_input_crop_encoded = le_crop.transform([user_input_crop])[0]

# Create a DataFrame with user input
user_input_df = pd.DataFrame([[user_input_crop_year, user_input_district_encoded, user_input_season_encoded, user_input_crop_encoded, user_input_area]],
                              columns=['Crop_Year', 'District_Name', 'Season', 'Crop', 'Area'])

# Make prediction using XGBoost for user input
user_input_prediction = xgb_regr.predict(user_input_df)

print(f"Predicted Production for the given input: {user_input_prediction[0]}")
