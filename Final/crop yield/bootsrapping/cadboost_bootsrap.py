import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the number of bootstrap samples
num_bootstraps = 10

# Initialize lists to store R-squared values for training and test sets
train_r2_scores = []
test_r2_scores = []

# Perform bootstrapping
for _ in range(num_bootstraps):
    # Create a bootstrap sample
    X_train_bootstrap, y_train_bootstrap = resample(X_train, y_train, replace=True, random_state=42)

    # CatBoost model for regression with increased complexity (intentional overfitting)
    cat_regr_bootstrap = CatBoostRegressor(iterations=1000, depth=10, learning_rate=0.01, random_state=42)

    # Train the overfit CatBoost model on the bootstrap sample
    cat_regr_bootstrap.fit(X_train_bootstrap, y_train_bootstrap, cat_features=['District_Name', 'Season', 'Crop'])

    # Make predictions on the original training set
    y_pred_train_bootstrap = cat_regr_bootstrap.predict(X_train)
    train_r2_scores.append(r2_score(y_train, y_pred_train_bootstrap))

    # Make predictions on the test set
    y_pred_test_bootstrap = cat_regr_bootstrap.predict(X_test)
    test_r2_scores.append(r2_score(y_test, y_pred_test_bootstrap))

# Calculate and print the mean and standard deviation of R-squared scores
mean_train_r2 = np.mean(train_r2_scores)
std_train_r2 = np.std(train_r2_scores)
mean_test_r2 = np.mean(test_r2_scores)
std_test_r2 = np.std(test_r2_scores)

print("Mean Training R-squared:", mean_train_r2)
print("Standard Deviation Training R-squared:", std_train_r2)
print("Mean Test R-squared:", mean_test_r2)
print("Standard Deviation Test R-squared:", std_test_r2)

# Optionally, you can print accuracy as well
accuracy_train = cat_regr_bootstrap.score(X_train, y_train)
accuracy_test = cat_regr_bootstrap.score(X_test, y_test)

print("Training Accuracy (Last Bootstrap Sample):", accuracy_train)
print("Test Accuracy (Last Bootstrap Sample):", accuracy_test)
