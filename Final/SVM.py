import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset into a pandas DataFrame
df1 = pd.read_csv("D:/EDAI/processed_apy_maharashtra.csv")

X = df1[['Crop_Year', 'District_Name', 'Season', 'Crop', 'Area']]
y = df1['Production']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM model for regression
svm_regr = SVR(kernel='linear')  # You can choose different kernels (linear, rbf, etc.)

# Train the model
svm_regr.fit(X_train, y_train)

# Make predictions
y_pred_train_svm = svm_regr.predict(X_train)
y_pred_test_svm = svm_regr.predict(X_test)

# Evaluate the model
print("Training Accuracy (SVM):", r2_score(y_train, y_pred_train_svm))
print("Test Accuracy (SVM):", r2_score(y_test, y_pred_test_svm))