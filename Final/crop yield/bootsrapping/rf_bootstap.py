import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

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

# Random Forest model for regression
rf_regr = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_regr.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_regr.feature_importances_

# Visualize feature importances
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.show()

# Select features based on importance (e.g., top 3 features)
selected_features = feature_names[sorted_idx][-3:]
print("Selected Features:", selected_features)

# Set the number of bootstrap samples
num_bootstrap_samples = 10

# Initialize lists to store in-sample and out-of-sample scores
in_sample_scores = []
out_of_sample_scores = []

# Perform bootstrapping
for i in range(num_bootstrap_samples):
    # Create a bootstrap sample
    X_bootstrap, y_bootstrap = resample(X_train[selected_features], y_train, random_state=i)

    # Train a Random Forest model on the bootstrap sample
    rf_regr_bootstrap = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regr_bootstrap.fit(X_bootstrap, y_bootstrap)

    # Evaluate the model on the in-sample data
    y_pred_train_bootstrap = rf_regr_bootstrap.predict(X_bootstrap)
    in_sample_scores.append(r2_score(y_bootstrap, y_pred_train_bootstrap))

    # Evaluate the model on the out-of-sample data
    y_pred_test_bootstrap = rf_regr_bootstrap.predict(X_test[selected_features])
    out_of_sample_scores.append(r2_score(y_test, y_pred_test_bootstrap))

# Calculate and print the average in-sample and out-of-sample scores
average_in_sample_score = np.mean(in_sample_scores)
average_out_of_sample_score = np.mean(out_of_sample_scores)

print("Average In-Sample Score:", average_in_sample_score)
print("Average Out-of-Sample Score:", average_out_of_sample_score)
