# PREPROCESSING

# 1. Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

# 2. Loading Data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'CO2 Emissions.csv'
file_path = os.path.join(script_dir, file_name)
data = pd.read_csv(file_path)

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


# TRAINING USING L1 REGULARIZATION

# Creating and train the Lasso (L1) regression model
lasso_model = Lasso(alpha=1.0)  
lasso_model.fit(X_train, y_train)

# Making predictions using the Lasso model
y_test_pred_lasso = lasso_model.predict(X_test)

# Calculating evaluation metrics for Lasso (L1) regression
mse_lasso = mean_squared_error(y_test, y_test_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_test_pred_lasso)
n_lasso, p_lasso = X_test.shape[0], X_test.shape[1]
adj_r2_lasso = 1 - (1 - r2_lasso) * ((n_lasso - 1) / (n_lasso - p_lasso - 1))
mae_lasso = mean_absolute_error(y_test, y_test_pred_lasso)

# TRAINING USING L2 REGULARIZATION

# Creating and training the Ridge (L2) regression model
ridge_model = Ridge(alpha=1.0)  
ridge_model.fit(X_train, y_train)

# Making predictions using the Ridge model
y_test_pred_ridge = ridge_model.predict(X_test)

# Calculating evaluation metrics for Ridge (L2) regression
mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_test_pred_ridge)
n_ridge, p_ridge = X_test.shape[0], X_test.shape[1]
adj_r2_ridge = 1 - (1 - r2_ridge) * ((n_ridge - 1) / (n_ridge - p_ridge - 1))
mae_ridge = mean_absolute_error(y_test, y_test_pred_ridge)

# Print the results for both models
print("Metrics for Lasso (L1) Regression:")
print(f"MSE: {mse_lasso:.2f}")
print(f"RMSE: {rmse_lasso:.2f}")
print(f"R2 Score: {r2_lasso:.2f}")
print(f"Adjusted R2 Score: {adj_r2_lasso:.2f}")
print(f"MAE: {mae_lasso:.2f}")
print("\n")

print("Metrics for Ridge (L2) Regression:")
print(f"MSE: {mse_ridge:.2f}")
print(f"RMSE: {rmse_ridge:.2f}")
print(f"R2 Score: {r2_ridge:.2f}")
print(f"Adjusted R2 Score: {adj_r2_ridge:.2f}")
print(f"MAE: {mae_ridge:.2f}")