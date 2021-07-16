from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import json
from LSTM import LstmModel
# Setup flask server
app = Flask(__name__)
CORS(app)
# Setup url route which will calculate
# total sum of array.

models = None

@app.route('/model', methods=['GET'])
def sum_of_array():
    model = request.args.get('model')
    if model == 'LSTM':
        result = models.getSimpleResult()
    else:
        result = "HELELELE"
    # Return data in json format
    return json.dumps(result)


if __name__ == "__main__":

    df = pd.read_csv("NSE-TATA.csv")
    df.head()

    #Analyze the closing prices from dataframe:+
    df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
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
    models = LstmModel(new_dataset)
    app.run(port=5000)
