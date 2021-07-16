import utils as utils
import numpy as np

class SvrModel(object):
    def __init__(self, dataset):
        super().__init__()
        dataset_value = dataset.values
        train_data,test_data = utils.create_data(dataset_value,0.8)
        number_of_previous_close_prices = 60
        x_test, _ = utils.create_x_y_test(test_data,number_of_previous_close_prices)
        regressor = utils.load_pickle("SVR.pkl")
        y_pred = regressor.predict(x_test)

        num_test = len(dataset_value) - len(x_test)
        self.indexs = dataset.index[num_test:]
        self.predictions = y_pred
        self.close = dataset['Close'].values[num_test:]
        print("SVR init success")

    def getSimpleResult(self):
        predictions = []
        close = []
        indexs = []
        for index in range(0, len(self.indexs)):
            predictions.append(np.float64(self.predictions[index].item()))
            close.append(self.close[index])
            indexs.append(self.indexs[index].value // 1000000)
        return {'Predictions' : predictions, 'Close': close, 'Index': indexs}