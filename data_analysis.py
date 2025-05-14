# data_analysis.py

# Task 1: Load and Explore the Dataset
import pandas as pd
from sklearn.datasets import load_iris

print("Loading dataset...")
# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check the data types and any missing values
print("Dataset info and description:")
print(df.info())
print(df.describe(include='all'))

# Check for missing values
print("Missing values:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis
# Compute basic statistics of the numerical columns
print("Basic statistics of numerical columns:")
print(df.describe())

# Group by 'species' and compute the mean of numerical columns
group_by_species = df.groupby('species').mean()
print("Grouped data by species:")
print(group_by_species)

# Task 3: Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("Setting up matplotlib...")
# Ensure matplotlib is using an appropriate backend for scripts
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend to save figures to files

# Line chart
plt.figure(figsize=(10, 6))
sns.lineplot(data=group_by_species, marker='o')
plt.title('Average Sepal and Petal Measurements by Species')
plt.xlabel('Species')
plt.ylabel('Average Measurement (cm)')
plt.legend(title='Measurement')
plt.savefig('line_chart.png')
plt.close()

# Bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.savefig('bar_chart.png')
plt.close()

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['petal length (cm)'], kde=True)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.savefig('histogram.png')
plt.close()

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.savefig('scatter_plot.png')
plt.close()

print("Plots have been saved as PNG files.")