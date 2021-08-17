import utils
import numpy as np
n_period_before = 10
class LinearModel(object):
    def __init__(self, dataset):
        super().__init__()
        dataset_value = dataset.values
        _, test_data = utils.create_data(dataset_value, 0.8)
        number_of_previous_close_prices = 60
        x_test, y_test = utils.create_x_y_test(
            test_data, number_of_previous_close_prices)

        model2 = utils.load_pickle("linear.pkl")
        y_pred2 = model2.predict(x_test)

        num_test = len(dataset_value) - len(x_test)
        self.indexs = dataset.index[num_test:]
        self.predictions = y_pred2
        self.close = dataset['Close'].values[num_test:]
        
        price_of_change = y_pred2[n_period_before:]
        for i in range(0, len(price_of_change)):
            price_of_change[i] = (price_of_change[i]-y_pred2[i])/y_pred2[i]
        self.poc = price_of_change

        print("Linear init success")

    def getSimpleResult(self):
        predictions = []
        close = []
        indexs = []
        for index in range(0, len(self.indexs)):
            predictions.append(np.float64(self.predictions[index].item()))
            close.append(self.close[index])
            indexs.append(self.indexs[index].value // 1000000)
        return {'Predictions' : predictions, 'Close': close, 'Index': indexs, 'poc': self.poc, 'IndexPoC': indexs[n_period_before:]}