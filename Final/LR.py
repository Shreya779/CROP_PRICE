#POLYNOMAIL REGRESSION

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset into a pandas DataFrame
df1 = pd.read_csv("D:/EDAI/processed_apy_maharashtra.csv")

# Assuming 'District_Name', 'Season', and 'Crop' are categorical variables
categorical_features = ['District_Name', 'Season', 'Crop']
numeric_features = ['Crop_Year', 'Area']

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Define the polynomial regression model
degree = 2  # Adjust the degree as needed
polynomial_regr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('polynomial_features', PolynomialFeatures(degree=degree)),
    ('regressor', LinearRegression())
])

X = df1[['Crop_Year', 'District_Name', 'Season', 'Crop', 'Area']]
y = df1['Production']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model using the pipeline
polynomial_regr.fit(X_train, y_train)

# Continue with model evaluation
y_pred_train_poly = polynomial_regr.predict(X_train)
y_pred_test_poly = polynomial_regr.predict(X_test)

print("Training Accuracy (R2 Score):", r2_score(y_train, y_pred_train_poly))
print("Test Accuracy (R2 Score):", r2_score(y_test, y_pred_test_poly))
