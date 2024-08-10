import numpy as np
import lightgbm as lgb
import joblib

preprocessor = joblib.load("preprocessor.joblib")

# Load preprocessed data
data = np.load('processed_data.npz')
X_train = data['X_train']
y_train = data['y_train']

feature_names = preprocessor.get_feature_names_out()
print(feature_names.shape)
print(X_train.shape)

# Initialize and train the LightGBM model
lgb_train = lgb.Dataset(X_train, y_train,feature_name=feature_names.tolist())
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'n_estimators': 1000
}
model = lgb.train(params, lgb_train, valid_sets=[lgb_train], callbacks=[lgb.early_stopping(stopping_rounds=5)])

# Save the trained model
model.save_model('lgbm_model.txt',num_iteration=model.best_iteration)

lgb.cv(params, train_data, num_round, nfold=5)

