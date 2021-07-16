from flask import Flask, request
from flask_cors import CORS
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
    models = LstmModel()
    app.run(port=5000)
