import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset into a pandas DataFrame
df = pd.read_csv("D:/EDAI/maharashtra.csv")

# Impute missing values using IterativeImputer
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

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural Network model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with more epochs
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions using the trained model
y_pred_train_nn = model.predict(X_train)
y_pred_test_nn = model.predict(X_test)

# Evaluate the neural network model
print("Training Accuracy (Neural Network):", r2_score(y_train, y_pred_train_nn))
print("Test Accuracy (Neural Network):", r2_score(y_test, y_pred_test_nn))
