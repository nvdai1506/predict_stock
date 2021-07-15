
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
# from config import LSTM_MODEL_OUTPUT

CSV = "NSE-TATA.csv"
MODEL = "LSTM.h5"
class LstmModel(object):
    def __init__(self):
        super().__init__()
        df_nse = pd.read_csv(CSV)
        df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
        df_nse.index = df_nse['Date']

        new_data = pd.DataFrame(index=range(
            0, len(df_nse)), columns=['Date', 'Close'])
        data = df_nse.sort_index(ascending=True, axis=0)

        for i in range(0, len(data)):
            new_data["Date"][i] = data['Date'][i]
            new_data["Close"][i] = data["Close"][i]

        new_data.index = new_data.Date
        new_data.drop("Date", axis=1, inplace=True)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(new_data.values)

        inputs = new_data[len(new_data) -
                          len(new_data.values[987:, :])-60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(60, inputs.shape[0]):
            X_test.append(inputs[i-60:i, 0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        model = load_model(MODEL)
        closing_price = model.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)
        self.valid = new_data[987:]
        self.valid['Predictions'] = closing_price

    def getResult(self):
        return self.valid

    def getSimpleResult(self):
        result = []
        for index in range(0, len(self.valid)):
            result.append((self.valid.index[index].value / 1000000, self.valid.values[index][0]))
        return result
