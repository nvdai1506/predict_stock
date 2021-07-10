import numpy as np
import pandas as pd

import utils as u
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sklearn.model_selection import  GridSearchCV


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



model = SVR(kernel = 'linear')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Params: ",model.get_params())
print("Score: ", model.score(x_train,y_train))
rmse = np.sqrt(np.mean(y_pred - y_test)**2)
print('rmse: ', rmse)





# parameters = [    
#         {
#             'C':np.logspace(-4, 4, 10), #Regularization parameter. The strength of the regularization is inversely proportional to C.
#             'gamma': ['scale', 'auto'], #Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’
#             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'] #specifies the kernel type to be used in the algorithm
#         }
#     ]

# grid_search = GridSearchCV(estimator = SVR(), param_grid = parameters , scoring = 'accuracy', cv = 3, n_jobs = -1 , verbose = 2)
# grid_scores = grid_search.fit(x_train , y_train)
# print( grid_search)
# print( grid_search.best_score_)
# print(grid_search.best_params_)

# u.save_model(grid_search,'SVR.txt')

# params = u.load_model('SVR.txt')
# u.SVR_model(x_train,y_train,x_test,y_test,params)