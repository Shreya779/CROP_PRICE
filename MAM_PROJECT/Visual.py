# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Load the dataset
# data = pd.read_csv('Crop_recommendation.csv')
#
# # Drop rows with missing labels
# data = data.dropna(subset=['label'])
#
# # Count the occurrences of each label
# label_counts = data['label'].value_counts()
#
# # Create a pie chart
# plt.figure(figsize=(8, 6))
# plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%')
# plt.title('Distribution of Crop Labels', fontsize=16)
# plt.axis('equal')  # Equal aspect ratio ensures the pie chart is drawn as a circle
# plt.show()




import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Drop rows with missing labels
data = data.dropna(subset=['label'])

# Separate numerical and categorical columns
numerical_cols = ['Nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
categorical_cols = ['label']

# Create a correlation matrix for numerical columns
corr_matrix = data[numerical_cols].corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap', fontsize=16)
plt.show()