import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
# %matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense


from sklearn.preprocessing import MinMaxScaler
import utils as u
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

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

# n previous close price wanna traning
number_of_previous_close_prices = 60

train_data,test_data = u.create_data(scaled_data,0.8)



x_train, y_train = u.create_x_y_train(train_data,number_of_previous_close_prices)
x_test, y_test = u.create_x_y_test(test_data,number_of_previous_close_prices)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


#Build and train the LSTM model:
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train,y_train,epochs=2,batch_size=1,verbose=2)


#Take a sample of a dataset to make stock price predictions using the LSTM model:

x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predictions=lstm_model.predict(x_test)
# predictions=scaler.inverse_transform(predictions)



#get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print('rmse: ', rmse)

# Save the LSTM model:
lstm_model.save("LSTM.h5")

#Visualize the predicted stock costs with actual stock costs:
# train_data=new_dataset[:training_data_len]
# valid_data=new_dataset[training_data_len:]
# valid_data['Predictions']=predictions
# plt.plot(train_data["Close"])
# plt.plot(valid_data[['Close',"Predictions"]])