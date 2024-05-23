import cudf
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

nvda_data = cudf.read_csv('NVDA.csv')

nvda_data.isna().sum()

print(nvda_data.head())

print(nvda_data.describe())

nvda_data['Date'] = cudf.to_datetime(nvda_data['Date'])

nvda_data = nvda_data.dropna()

print(nvda_data.head())

mean_volume = nvda_data['Volume'].mean()
std_volume = nvda_data['Volume'].std()
z_score_threshold = 3
nvda_data = nvda_data[(nvda_data['Volume'] >= mean_volume - z_score_threshold * std_volume) &
                      (nvda_data['Volume'] <= mean_volume + z_score_threshold * std_volume)]

nvda_data['DailyChange'] = nvda_data['Close'] - nvda_data['Open']
nvda_data['PriceRange'] = nvda_data['High'] - nvda_data['Low']
nvda_data['VolumeChange'] = nvda_data['Volume'].diff()
nvda_data['VolumeRange'] = nvda_data['Volume'] / nvda_data['Volume'].shift(1)

print(nvda_data.head())

print(nvda_data.info())

print("Summary Statistics:")
print(nvda_data.describe())

df_pandas = nvda_data.to_pandas()

numeric_features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
df_pandas[numeric_features].hist(bins=20, figsize=(12, 8))
plt.suptitle("Distribution of Numeric Features", y=1.02)
plt.tight_layout()
plt.show()

correlation_matrix = nvda_data[numeric_features].corr().to_numpy()
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar().set_label('Correlation Coefficient')
plt.title('Correlation Matrix of Numeric Features')
plt.xticks(range(len(numeric_features)), numeric_features, rotation=45)
plt.yticks(range(len(numeric_features)), numeric_features)
plt.tight_layout
plt.show()

nvda_data['VolumeChange'] = nvda_data['VolumeChange'].astype('float64')

numeric_columns = ['DailyChange', 'PriceRange', 'VolumeChange', 'VolumeRange']
nvda_data[numeric_columns] = nvda_data[numeric_columns].fillna(nvda_data[numeric_columns].mean())

X = nvda_data[numeric_columns].to_cupy()
y = nvda_data['Close'].to_cupy()

X = cp.asnumpy(X)
y = cp.asnumpy(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
feature_importances = rf.feature_importances_
indices = feature_importances.argsort()[::-1]
plt.figure(figsize=(8, 6))
plt.bar(range(X_train.shape[1]), feature_importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), [numeric_columns[i] for i in indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance Ranking')
plt.tight_layout()
plt.show()

lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)
lasso_coef_ = lasso.coef_
selected_features_lasso = [numeric_columns[i] if lasso_coef_[i] != 0 else None for i in range(len(numeric_columns))]
selected_features_lasso = [feature for feature in selected_features_lasso if feature is not None]
print("Selected Features (Lasso):", selected_features_lasso)

y_pred = rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2 ): {r2}")

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values', color='blue')
plt.plot(y_pred, label='Predicted Values', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()

param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth' : [None, 10, 20],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=5, scoring = 'neg_mean_squared_error',
                           verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("best parameters: ", best_params)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ", mse)
print("R2 Score:", r2)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values', color='blue')
plt.plot(y_pred, label='Predicted Values', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()

param_grid = {
    'n_estimators' : [50, 100, 200],
    'learning_rate' : [0.01, 0.1, 0.5],
    'max_depth' : [3, 5, 10],
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4]
}

gbr = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid,
                           cv=5, scoring = 'neg_mean_squared_error',
                           verbose=2, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("best parameters: ", best_params)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error: ", mse)
print("R2 Score:", r2)

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Values', color='blue')
plt.plot(y_pred, label='Predicted Values', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Stock Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.show()