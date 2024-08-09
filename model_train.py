import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Load preprocessed data
data = np.load('processed_data.npz')
X_train = data['X_train']
y_train = data['y_train']

# Initialize and train the LightGBM model
lgb_train = lgb.Dataset(X_train, y_train)
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

lgb.cv(param, train_data, num_round, nfold=5)

