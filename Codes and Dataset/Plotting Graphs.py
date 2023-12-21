import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'CO2 Emissions.csv'
file_path = os.path.join(script_dir, file_name)
df = pd.read_csv(file_path)



# Scatter Plots
sns.pairplot(df, diag_kind='kde')
plt.show()

# Box Plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# Correlation Heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Distribution Plots (Histograms for numerical features)
numerical_features = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12, 8))
for i, column in enumerate(numerical_features.columns):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[column], kde=True)
    plt.title(column)
plt.tight_layout()
plt.show()

# Pie Charts (for categorical features)
categorical_features = df.select_dtypes(include=['object'])
plt.figure(figsize=(12, 8))
for i, column in enumerate(categorical_features.columns):
    plt.subplot(2, 2, i+1)
    category_counts = df[column].value_counts()
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(column)
plt.tight_layout()
plt.show()




