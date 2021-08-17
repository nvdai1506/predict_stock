
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import utils
# from config import LSTM_MODEL_OUTPUT

CSV = "NSE-TATA.csv"
MODEL = "LSTM.h5"
n_period_before = 10
class LstmModel(object):
    def __init__(self, dataset):
        super().__init__()

        dataset_values = dataset.values
        # n previous close price wanna traning
        number_of_previous_close_prices = 60

        scaler=MinMaxScaler(feature_range=(0,1))
        scaled_data=scaler.fit_transform(dataset_values)
        train_data,test_data = utils.create_data(scaled_data,0.8, number_of_previous_close_prices)
        x_test, _ = utils.create_x_y_test(test_data,number_of_previous_close_prices)

        lstm_model = load_model(MODEL)
        x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
        closing_price=lstm_model.predict(x_test)
        closing_price=scaler.inverse_transform(closing_price)
        num_test = len(scaled_data) - len(x_test)
        self.indexs = dataset.index[num_test:]
        self.predictions = closing_price
        self.close = dataset['Close'].values[num_test:]
        
        price_of_change = closing_price[n_period_before:]
        poc = []
        for i in range(0, len(price_of_change)):
            value = (price_of_change[i]-closing_price[i])/closing_price[i]
            poc.append(np.float64(value.item()))
        self.poc = poc

        print("LSTM init success")
    def getResult(self):
        return self.valid

    def getSimpleResult(self):
        predictions = []
        close = []
        indexs = []
        for index in range(0, len(self.indexs)):
            predictions.append(np.float64(self.predictions[index].item()))
            close.append(self.close[index])
            indexs.append(self.indexs[index].value // 1000000)
        return {'Predictions' : predictions, 'Close': close, 'Index': indexs, 'poc': self.poc, 'IndexPoC': indexs[n_period_before:]}
