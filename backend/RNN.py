
import numpy as np
from os import path
from utils import load_RNN

n_windows = 10
n_input = 1
n_output = 1
size_train = 1001
n_period_before = 10
class RnnModel(object):
    def __init__(self, dataset):
        super().__init__()
        series = np.array(dataset.values)
        train = series[:size_train]
        test = series[size_train:1232]
        _, _, x_test, y_test = self.create_batches(train=train, test=test,windows=n_windows,
                                                      input=n_input,
                                                      output=n_output)

        y_pred = load_RNN("RNN.ckpt.meta", x_test, y_test)
        y_pred = y_pred.flatten()
        self.indexs = dataset.index[size_train:]
        self.predictions = y_pred
        self.close = dataset['Close'].values[size_train:]
        
        price_of_change = y_pred[n_period_before:]
        poc = []
        for i in range(0, len(price_of_change)):
            value = (price_of_change[i]-y_pred[i])/y_pred[i]
            poc.append(np.float64(value.item()))
        self.poc = poc

        print("RNN init success")

    def create_batches(self, train, test, windows, input, output):
    ## Create X
        x_data = train[:size_train-1]  # Select the data
        x_batches = x_data.reshape(-1, windows, input)  # Reshape the data
        x_test_data = test[:len(test)-1]
        x_test = x_test_data.reshape(-1, windows, input)  # Reshape the data

        ## Create y
        y_data = train[n_output:size_train]
        y_batches = y_data.reshape(-1, windows, output)
        y_test_data = test[n_output:]
        y_test = y_test_data.reshape(-1, windows, output)
        return x_batches, y_batches, x_test, y_test

    def getSimpleResult(self):
        predictions = []
        close = []
        indexs = []
        for index in range(0, len(self.predictions)):
            predictions.append(np.float64(self.predictions[index].item()))
            close.append(self.close[index])
            indexs.append(self.indexs[index].value // 1000000)
        return {'Predictions' : predictions, 'Close': close, 'Index': indexs, 'poc': self.poc, 'IndexPoC': indexs[n_period_before:]}