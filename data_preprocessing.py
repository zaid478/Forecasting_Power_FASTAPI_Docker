import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import joblib

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['temp_t+1', 'feels_like_t+1'])

    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Create lag features for 'active_power'
    df['active_power_lag_1'] = df['active_power'].shift(1)
    df['active_power_lag_2'] = df['active_power'].shift(2)
    df['active_power_rolling_mean'] = df['active_power'].rolling(window=3).mean()

    # Handle missing values by forward filling
    df.fillna(method='ffill', inplace=True)

    # Define features and target
    X = df.drop(columns=['active_power', 'date','main'])
    y = df['active_power']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define a preprocessing pipeline
    categorical_features = ['description']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Preprocess the data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    joblib.dump(preprocessor, 'preprocessor.joblib')


    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    np.savez('processed_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Data preprocessing completed.")
