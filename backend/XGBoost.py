import utils
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

class XgBoostModel(object):
    def __init__(self, df):
        super().__init__()
        x_train, y_train, x_test, y_test, x_valid, y_valid, indexs = utils.XGBoost_preprocessing_data(df)
        eval_set = [(x_train, y_train), (x_valid, y_valid)]
        clf2 = utils.load_model('XGBoost')
        model2 = xgb.XGBRegressor(**clf2.best_params_, objective='reg:squarederror')
        model2.fit(x_train, y_train, eval_set=eval_set, verbose=False)
        ## Calculate and visualize predictions
        y_pred2 = model2.predict(x_test)
        print(f'y_pred = {y_pred2[:5]}')
        print(f'y_test = {y_test[:5]}')
        self.indexs = indexs
        self.predictions = y_pred2
        self.close = y_test
        print(f'mean_squared_error = {mean_squared_error(y_test, y_pred2)}')
        print("SVR init success")

    def getSimpleResult(self):
        predictions = []
        close = []
        indexs = []
        for index in range(0, len(self.indexs)):
            predictions.append(np.float64(self.predictions[index].item()))
            close.append(self.close.values[index])
            indexs.append(self.indexs.values[index].item() // 1000000)
        return {'Predictions' : predictions, 'Close': close, 'Index': indexs}