import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset into a pandas DataFrame
df1 = pd.read_csv("D:/EDAI/apy.csv")

# Filter data for Maharashtra
df = df1[df1['State_Name'] == 'Maharashtra']

# Drop the 'State_Name' column
df = df.drop('State_Name', axis=1)

# Save the DataFrame to a new CSV file
df.to_csv("D:/EDAI/maharashtra.csv", index=False)
