# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a pandas DataFrame
file_path = 'data.csv'  # Replace with the actual file path
df = pd.read_csv(file_path)

# 1. Display the first few rows of the dataframe
print(df.head())

# 2. Get a summary of the dataset (column names, data types, and non-null counts)
print(df.info())

# 3. Statistical summary of numerical columns
print(df.describe())

# 4. Check for missing values
missing_data = df.isnull().sum()
print("Missing data per column:\n", missing_data)

# 5. Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")

# 6. Data Distribution (if numerical data is present)
numerical_columns = df.select_dtypes(include=[np.number]).columns
for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

# 7. Check correlations (for numerical columns)
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 8. Check relationships between variables using pairplot (if numerical)
sns.pairplot(df[numerical_columns])
plt.show()

# 9. Boxplots to detect outliers (for numerical columns)
for column in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot of {column}')
    plt.show()

# 10. Check for categorical variables (if any)
categorical_columns = df.select_dtypes(include=[object]).columns
for column in categorical_columns:
    print(f"Value counts for {column}:\n", df[column].value_counts())

# 11. Visualize categorical data distribution (if any categorical columns)
for column in categorical_columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=df[column])
    plt.title(f'Distribution of {column}')
    plt.show()
