import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset into a pandas DataFrame
df1 = pd.read_csv("D:/EDAI/processed_apy_maharashtra.csv")

# Assuming 'District_Name', 'Season', and 'Crop' are categorical variables
categorical_features = ['District_Name', 'Season', 'Crop']
numeric_features = ['Crop_Year', 'Area']

# Convert categorical features to the 'category' dtype
df1[categorical_features] = df1[categorical_features].astype('category')

# Encode categorical variables using LabelEncoder
label_encoder = LabelEncoder()
for feature in categorical_features:
    df1[feature] = label_encoder.fit_transform(df1[feature])

X = df1[['Crop_Year', 'District_Name', 'Season', 'Crop', 'Area']]
y = df1['Production']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# K-Nearest Neighbors Regressor model
knn_regressor = KNeighborsRegressor(n_neighbors=7)  # You can adjust the number of neighbors

# Train the model
knn_regressor.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train_knn = knn_regressor.predict(X_train_scaled)
y_pred_test_knn = knn_regressor.predict(X_test_scaled)

# Evaluate the model
print("Training Accuracy (KNN):", r2_score(y_train, y_pred_train_knn))
print("Test Accuracy (KNN):", r2_score(y_test, y_pred_test_knn))
