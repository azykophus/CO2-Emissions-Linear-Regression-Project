# PREPROCESSING

# 1. Importing Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# 2. Loading Data
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'CO2 Emissions.csv'
file_path = os.path.join(script_dir, file_name)
data = pd.read_csv(file_path)

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

# PERFORMING LINEAR REGRESSION ON THE PREPROCESSED DATA

# 1. Creating and training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 2. Making predictions
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)


# 3. Evaulating the Model

# Mean Squared Error
mse_test = mean_squared_error(y_test, y_pred_test) 
mse_train = mean_squared_error(y_train, y_pred_train) 

# Root Mean Squared Error
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# R2 Score
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Adjsuted R2 Score
n_test = len(y_test)
p_test = X_test.shape[1]

n_train = len(y_train)
p_train = X_train.shape[1]

adjusted_r2_train = 1 - ((1 - r2_train) * (n_train - 1) / (n_train - p_train - 1))
adjusted_r2_test = 1 - ((1 - r2_test) * (n_test - 1) / (n_test - p_test - 1))

# Mean Absolute Error
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("Metrics for the Training Data:")
print(f"MSE: {mse_train:.2f}")
print(f"RMSE: {rmse_train:.2f}")
print(f"R2 Score: {r2_train:.2f}")
print(f"Adjusted R2 Score: {adjusted_r2_train:.2f}")
print(f"MAE: {mae_train:.2f}")

print("\nMetrics for the Test Data:")
print(f"MSE: {mse_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R2 Score: {r2_test:.2f}")
print(f"Adjusted R2 Score: {adjusted_r2_test:.2f}")
print(f"MAE: {mae_test:.2f}")



