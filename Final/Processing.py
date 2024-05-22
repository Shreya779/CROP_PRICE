import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset into a pandas DataFrame
df1 = pd.read_csv("D:/EDAI/apy.csv")
print(df1.isnull().sum())
df = df1.dropna(axis=0)  # axis=0 refers to rows
print(df.isnull().sum())

# Filter data for the state 'Maharashtra'
df_filtered = df[df['State_Name'] == 'Maharashtra']

# Drop the 'State_Name' column
df_filtered = df_filtered.drop('State_Name', axis=1)

# Encode categorical variables
le_district = LabelEncoder()
df_filtered['District_Name'] = le_district.fit_transform(df_filtered['District_Name'])
le_season = LabelEncoder()
df_filtered['Season'] = le_season.fit_transform(df_filtered['Season'])
le_crop = LabelEncoder()
df_filtered['Crop'] = le_crop.fit_transform(df_filtered['Crop'])

# Save the processed DataFrame to a new CSV file
df_filtered.to_csv("D:/EDAI/processed_apy_maharashtra.csv, index=False")
