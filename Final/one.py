import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv("D:/EDAI/apy.csv")

print("Count of missing values in each column:")
# print(df.isnull().sum())

print("\nInformation about the DataFrame:")
# print(df.info())

print("\nCount of duplicated rows:", df.duplicated().sum())

df_filtered = df[df['State_Name'] == 'Maharashtra']

# Drop the 'State_Name' column
df_filtered = df_filtered.drop('State_Name', axis=1)
# print(df_filtered.isnull().sum())
rf_regressor = RandomForestRegressor()

imputer = IterativeImputer(estimator=rf_regressor)

df['Production'] = imputer.fit_transform(df[['Production']])

print("\nMissing values after imputation:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_cols]

df_normalized = (df_numeric - df_numeric.min()) / (df_numeric.max() - df_numeric.min())

# Combine normalized numeric columns with non-numeric columns
for col in df.columns:
    if col not in numeric_cols:
        df_normalized[col] = df[col]

if 'Production' in df_normalized.columns:
    df_normalized.drop('Production', axis=1, inplace=True)

X = df_normalized
print(X.head())

df = pd.get_dummies(df)

# Separate the target variable (Yield) from the features
X = df.drop('Production', axis=1)
y = df['Production']

scaler = StandardScaler()

scaled_features = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_features, columns=X.columns)
print(scaled_df.head())

# Train the Random Forest Regressor model
rf_regressor.fit(X, y)

# Take input from the user for prediction
user_input_crop_year = int(input("Enter Crop Year: "))
user_input_season = input("Enter Season: ")
user_input_crop = input("Enter Crop: ")
user_input_area = float(input("Enter Area: "))

# Create a DataFrame with user input
user_input_df = pd.DataFrame([[user_input_crop_year, user_input_area]],
                              columns=['Crop_Year', 'Area'])

# Assume 'District_Name' and 'Season' are categorical columns
user_input_df['District_Name_Maharashtra'] = 1  # Set Maharashtra as the district
user_input_df[f'Season_{user_input_season}'] = 1
user_input_df[f'Crop_{user_input_crop}'] = 1

# Make prediction using Random Forest for user input
user_input_prediction = rf_regressor.predict(user_input_df)

print(f"Predicted Production for the given input: {user_input_prediction[0]}")
