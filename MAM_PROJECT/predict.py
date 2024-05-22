import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Display dataset information
df.info()
columns_to_remove = ['Unnamed: 8', 'Unnamed: 9']
df = df.drop(columns_to_remove, axis=1)
df.info()

# Extract features and labels
all_columns = df.columns[:-1]
label_encoder = LabelEncoder()
X = df[all_columns]
y = label_encoder.fit_transform(df["label"])
feature_names = all_columns.tolist()


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

with open('X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)

# RandomForest model
rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=18))
rf_pipeline.fit(X_train, y_train)
print("Success")

pickle.dump(rf_pipeline, open('C:/Users/shrey/PycharmProjects/pythonProject/MAM_PROJECT/model/rf_pipeline.sav', 'wb'))
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Run the Flask app
from app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)