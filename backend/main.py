from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import json
from LSTM import LstmModel
from RNN import RnnModel
from SVR import SvrModel
from XGBoost import XgBoostModel
from Linear import LinearModel

# Setup flask server
app = Flask(__name__)
CORS(app)
# Setup url route which will calculate
# total sum of array.

models = {}

@app.route('/model', methods=['GET'])
def sum_of_array():
    model = request.args.get('model')
    if model in models:
        result = models[model].getSimpleResult()
    else:
        result = "HELELELE"
    # Return data in json format
    return json.dumps(result)


if __name__ == "__main__":

    df = pd.read_csv("NSE-TATA.csv")
    df.head()

    #Analyze the closing prices from dataframe:+
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
    xgboost = XgBoostModel(df)
    df.index = df['Date']

    #Sort the dataset on date time and # filter “Date” and “Close” columns:
    data = df.sort_index(ascending=True, axis=0)

    new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_dataset["Date"][i] = data['Date'][i]
        new_dataset["Close"][i] = data["Close"][i]
    # Normalize the new filtered dataset:
    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis=1, inplace=True)

    rnn = RnnModel(new_dataset)
    models['RNN'] = rnn

    lstm = LstmModel(new_dataset)
    models['LSTM'] = lstm

    svr = SvrModel(new_dataset)
    models['SVR'] = svr

    models['XGBoost'] = xgboost

    linear = LinearModel(new_dataset)
    models['Linear'] = linear
    app.run(port=5000)
