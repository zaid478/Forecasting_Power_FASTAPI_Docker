import numpy as np
import lightgbm as lgb
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import mean_squared_error


# Load the trained model
model = lgb.Booster(model_file='lgbm_model.txt')

# Load and preprocess new data for inference
def preprocess_new_data(df, preprocessor):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df['active_power_lag_1'] = df['active_power'].shift(1)
    df['active_power_lag_2'] = df['active_power'].shift(2)
    df['active_power_rolling_mean'] = df['active_power'].rolling(window=3).mean()
    df.fillna(method='ffill', inplace=True)

    X = df.drop(columns=['active_power', 'date'])
    X = preprocessor.transform(X)

    return X

# Load preprocessor
data = np.load('processed_data.npz', allow_pickle=True)
preprocessor = data['preprocessor'][()]

# Load new data
# df_new = pd.read_csv("new_household_power_data.csv")
X_test = data['X_test']
y_test = data['y_test']


# Evaluate the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Test RMSE: {rmse}")

# # Preprocess the new data
# X_new = preprocess_new_data(df_new, preprocessor)

# # Make predictions
# predictions = model.predict(X_new)

# # Save the predictions
# df_new['predicted_active_power'] = predictions
# df_new.to_csv("predictions.csv", index=False)
# print("Predictions saved.")
