from fastapi import FastAPI
from pydantic import BaseModel
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib

# Define the input data model using Pydantic
class DataPoint(BaseModel):
    date: str
    current: float
    voltage: float
    reactive_power: float
    apparent_power: float
    power_factor: float
    main: str
    description: str
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    pressure: float
    humidity: float
    speed: float
    deg: float
    active_power_lag_1: float
    active_power_lag_2: float
    active_power_rolling_mean: float

# Initialize FastAPI
app = FastAPI()

# Load the trained model and preprocessor
model = lgb.Booster(model_file='lgbm_model.txt')
preprocessor = joblib.load('preprocessor.joblib')

# Preprocess a single data point
def preprocess_data_point(data: DataPoint):
    df = pd.DataFrame([data.dict()])
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # df['active_power_lag_1'] = np.nan  # Placeholders for lag features
    # df['active_power_lag_2'] = np.nan
    # df['active_power_rolling_mean'] = np.nan
    df.fillna(method='ffill', inplace=True)

    X = df.drop(columns=['date','main'])
    X = preprocessor.transform(X)

    return X

# Define the prediction endpoint
@app.post("/predict")
def predict(data: DataPoint):
    # Preprocess the input data point
    X_new = preprocess_data_point(data)

    # Make the prediction
    prediction = model.predict(X_new)

    # Return the prediction
    return {"predicted_active_power": prediction[0]}

# Run the FastAPI app with the command: uvicorn <filename>:app --reload
