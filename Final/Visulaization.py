import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import mplcursors

#Loading dataset
df1 = pd.read_csv("D:/EDAI/apy.csv")
# print(df1.isnull().sum())
#dropping missing values
df = df1.dropna(axis=0)  # axis=0 refers to rows
# print(df.isnull().sum())

#dataset for only maharashtra
df_filtered = df[(df['State_Name'] == 'Maharashtra')]
df_filtered = df_filtered.drop('State_Name', axis=1)

# plot for year vs production
plt.figure(figsize=(12, 6))
plt.bar(df_filtered['Crop_Year'], df_filtered['Production'], color='skyblue')
plt.xlabel('Year')
plt.ylabel('Production')
plt.title('Production vs Year for Maharashtra')
plt.show()



# fig = px.pie(df_filtered,
#              values='Production',
#              names='Crop',
#              title='Distribution of Yield Across Crops in Maharashtra',
#              hover_data=['Crop', 'Production'],  # Replace 'Yield' with a valid column name
#              labels={'Crop': 'Crops'},
#              hole=0.3)
#
# # Update layout for better appearance (optional)
# fig.update_traces(textinfo='none')  # Hide percentage labels
# fig.update_layout(showlegend=True)  # Show legend
#
# # Show the plot
# fig.show()


# fig = px.line(df_filtered, x='Crop_Year', y='Production', color='Crop',
#               labels={'Crop_Year': 'Year', 'Production': 'Production'},
#               title='Crop Production Over the Years in Maharashtra')
#
# # Show the plot
# fig.show()



# sns.lmplot(x='Area', y='Production', data=df, scatter_kws={'s': 20, 'alpha': 0.5})
#
# # Customize the plot
# plt.title('Scatter Plot with Trendline')
# plt.xlabel('Cultivated Area')
# plt.ylabel('Production')

# Show the plot
# plt.show()