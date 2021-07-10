import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

import utils as u
import statsmodels.api as sm


#Read the dataset:
df = pd.read_csv("NSE-TATA.csv")
df.head()

#Analyze the closing prices from dataframe:+
df["Date"] = pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index = df['Date']

#Sort the dataset on date time and # filter “Date” and “Close” columns:
data = df.sort_index(ascending=True,axis=0)
new_dataset = pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
for i in range(0,len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["Close"][i] = data["Close"][i]
# Normalize the new filtered dataset:
new_dataset.index = new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)
dataset = new_dataset.values

# n previous close price wanna traning
number_of_previous_close_prices = 60

train_data,test_data = u.create_data(dataset,0.8)



x_train, y_train = u.create_x_y_train(train_data,number_of_previous_close_prices)
x_test, y_test = u.create_x_y_test(test_data,number_of_previous_close_prices)

x_train = sm.add_constant(x_train)

x_train.reshape(-1,1)

y_train = np.array(y_train, dtype=float)
x_train = np.array(x_train, dtype=float)
model = sm.OLS(y_train, x_train)
results = model.fit()

print('coefficient of determination:', results.rsquared)
print('adjusted coefficient of determination:', results.rsquared_adj)
print('regression coefficients:', results.params)

x_test = sm.add_constant(x_test)
y_pred = results.predict(x_test)
rmse = np.sqrt(np.mean(y_pred - y_test)**2)
print('rmse: ', rmse)

results.save('Linear_statsmodels.txt')