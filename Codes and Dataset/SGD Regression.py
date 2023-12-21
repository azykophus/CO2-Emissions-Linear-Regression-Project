# PREPROCESSING

# 1. Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

# 2. Loading Data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'CO2 Emissions.csv'
file_path = os.path.join(script_dir, file_name)
data = pd.read_csv(file_path)
scaler = StandardScaler()

# 3. Checking for missing values
for i in data.isnull().sum():
    if i==0:
        pass
    else:
        print("Please handle missing values")
        break

# 4. Identify categorical features
categorical_features = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']

# 5. Applying label-based encoding to categorical features
label_encoder = LabelEncoder()

for feature in categorical_features:
    data[feature] = label_encoder.fit_transform(data[feature])

# 6. Splitting data into Training and Testing Sets
X = data.drop('CO2 Emissions(g/km)', axis=1)
y = data['CO2 Emissions(g/km)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TRAINING USING SGDRegressor MODEL

# Creating and training the SGDRegressor model for linear regression
sgd_model = SGDRegressor(max_iter=10000, tol=1e-3, random_state=42) 
sgd_model.fit(X_train_scaled, y_train)

# Making predictions using the SGDRegressor model
y_test_pred_sgd = sgd_model.predict(X_test_scaled)

# Calculating evaluation metrics for SGDRegressor
mse_sgd = mean_squared_error(y_test, y_test_pred_sgd)
rmse_sgd = np.sqrt(mse_sgd)
r2_sgd = r2_score(y_test, y_test_pred_sgd)
n_sgd, p_sgd = X_test.shape[0], X_test.shape[1]
adj_r2_sgd = 1 - (1 - r2_sgd) * ((n_sgd - 1) / (n_sgd - p_sgd - 1))
mae_sgd = mean_absolute_error(y_test, y_test_pred_sgd)

# Print the results for SGDRegressor
print("Metrics for SGDRegressor:")
print(f"MSE: {mse_sgd:.2f}")
print(f"RMSE: {rmse_sgd:.2f}")
print(f"R2 Score: {r2_sgd:.2f}")
print(f"Adjusted R2 Score: {adj_r2_sgd:.2f}")
print(f"MAE: {mae_sgd:.2f}")
