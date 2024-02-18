import pandas as pd
import plotly.express as px

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Load your dataset using pd.read_csv() or any other method
# Replace "your_dataset.csv" with the actual file path or data loading code
df = pd.read_csv("D:/edai/crop_yield.csv")
# print(df.isnull().sum())
# df.info()
# print(df.duplicated().sum())
# sns.scatterplot(x = df['Annual_Rainfall'], y = df['Yield'])
# plt.show()


# df_year = df[df['Crop_Year']!=2020]  # As the data of 2020 is incomplete
# year_yield = df_year.groupby('Crop_Year').sum()
# year_yield
# plt.figure(figsize = (12,5))
# plt.plot(year_yield.index, year_yield['Yield'],color='blue', linestyle='dashed', marker='o',
#         markersize=12, markerfacecolor='yellow')
# plt.xlabel('Year')
# plt.ylabel('Yield')
# plt.title('Measure of Yield over the year')
# plt.show()


# df_state = df.groupby('State').sum()
# df_state.sort_values(by = 'Yield', inplace=True, ascending = False)
# df_state['Region'] = ['States' for i in range(len(df_state))]
#
# fig = px.bar(df_state, x='Region', y = 'Yield', color=df_state.index, hover_data=['Yield'])
# fig.show()


df_state = df.groupby('State').sum()
df_state.sort_values(by='Yield', inplace=True, ascending=False)

fig = px.bar(df_state, x=df_state.index, y='Yield', color=df_state.index, hover_data=['Yield'])
fig.show()


# df_Seas = df[df['Season']!='Whole Year ']
#
# df_season = df_Seas.groupby('Season').sum()
# fig = px.sunburst(df_season, path=[df_season.index, 'Yield'], values='Yield',
#                   color=df_season.index, hover_data=['Yield'])
# fig.show()
# # Specify features (X) and target variable (y)
# # Replace 'target_column' with the actual name of your target column
# X = df.drop('Yield', axis=1)  # Features
# y = df['Yield']  # Target variable
#
# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Check the sizes of the subsets and the original dataset
# print("Original dataset size:", df.shape)
# print("Training set size:", X_train.shape, y_train.shape)
# print("Testing set size:", X_test.shape, y_test.shape)
