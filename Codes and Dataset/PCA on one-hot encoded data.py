# PREPROCESSING

# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import os

# 2. Loading Data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'CO2 Emissions.csv'
file_path = os.path.join(script_dir, file_name)
data = pd.read_csv(file_path)

# Checking for missing values
for i in data.isnull().sum():
    if i==0:
        pass
    else:
        print("Please handle missing values")
        break

# ENCODING THE CATEGORICAL FEATURES OF THE ORIGINAL DATASET WITH ONE-HOT EN-CODING

X = data.drop('CO2 Emissions(g/km)', axis=1)
y = data['CO2 Emissions(g/km)']
categorical_features = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)


# Splitting data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# PERFORMING PCA

# Defining the list of numbers of components to test
num_components_list = [10, 20, 50, 100, 200]

# Initialize dictionaries to store the evaluation metrics for each number of components
metrics_train = {}
metrics_test = {}

# Loop through different numbers of components
for num_components in num_components_list:
    # Apply PCA to reduce the number of features
    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train_pca, y_train)
    
    # Make predictions on the training and testing sets
    y_train_pred = model.predict(X_train_pca)
    y_test_pred = model.predict(X_test_pca)
    
    # Calculate evaluation metrics for training data
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_train_pred)
    n_train, p_train = X_train_pca.shape[0], X_train_pca.shape[1]
    adj_r2_train = 1 - (1 - r2_train) * ((n_train - 1) / (n_train - p_train - 1))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    
    # Calculate evaluation metrics for test data
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_test_pred)
    n_test, p_test = X_test_pca.shape[0], X_test_pca.shape[1]
    adj_r2_test = 1 - (1 - r2_test) * ((n_test - 1) / (n_test - p_test - 1))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    # Store metrics in dictionaries
    metrics_train[num_components] = {
        "MSE": mse_train,
        "RMSE": rmse_train,
        "R2": r2_train,
        "Adj_R2": adj_r2_train,
        "MAE": mae_train
    }
    
    metrics_test[num_components] = {
        "MSE": mse_test,
        "RMSE": rmse_test,
        "R2": r2_test,
        "Adj_R2": adj_r2_test,
        "MAE": mae_test
    }

# Print the results for different numbers of components on both train and test datasets
for num_components in num_components_list:
    print(f"Number of Components: {num_components}")
    
    print("Train Dataset:")
    train_metrics = metrics_train[num_components]
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.2f}")
    
    print("\nTest Dataset:")
    test_metrics = metrics_test[num_components]
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.2f}")
    
    print("\n")