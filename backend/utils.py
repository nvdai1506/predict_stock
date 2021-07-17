import pandas as pd
import numpy as np
import math
import joblib

from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

import tensorflow.compat.v1 as tf


def relative_strength_idx(df, n=14):
    close = df['Close']
    delta = close.diff()
    delta = delta[1:]
    pricesUp = delta.copy()
    pricesDown = delta.copy()
    pricesUp[pricesUp < 0] = 0
    pricesDown[pricesDown > 0] = 0
    rollUp = pricesUp.rolling(n).mean()
    rollDown = pricesDown.abs().rolling(n).mean()
    rs = rollUp / rollDown
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def XGBoost_preprocessing_data(df):
    #Decomposition
    df_close = df[['Date', 'Close']].copy()
    df_close = df_close.set_index('Date')    
    ##Moving Averages
    df['EMA_9'] = df['Close'].ewm(9).mean().shift()
    df['SMA_5'] = df['Close'].rolling(5).mean().shift()
    df['SMA_10'] = df['Close'].rolling(10).mean().shift()
    df['SMA_15'] = df['Close'].rolling(15).mean().shift()
    df['SMA_30'] = df['Close'].rolling(30).mean().shift()
    #Relative Strength Index
    df['RSI'] = relative_strength_idx(df).fillna(0)
    EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
    EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
    df['MACD'] = pd.Series(EMA_12 - EMA_26)
    df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

    # df['Close'] = df['Close'].shift(-1)
    df = df.iloc[33:] # Because of moving averages and MACD line
    df = df[:-1]      # Because of shifting close price

    df.index = range(len(df))
    test_size  = 0.15
    valid_size = 0.15

    test_split_idx  = int(df.shape[0] * (1-test_size))
    valid_split_idx = int(df.shape[0] * (1-(valid_size+test_size)))

    train_df  = df.loc[:valid_split_idx].copy()
    valid_df  = df.loc[valid_split_idx+1:test_split_idx].copy()
    test_df   = df.loc[test_split_idx+1:].copy()
    indexs = df.Date[test_split_idx+1:]
    #Drop unnecessary columns
    drop_cols = ['Date', 'Open', 'Low', 'High']
    train_df = train_df.drop(drop_cols, 1)
    valid_df = valid_df.drop(drop_cols, 1)
    test_df  = test_df.drop(drop_cols, 1)
    #Split into features and labels
    y_train = train_df['Close'].copy()
    X_train = train_df.drop(['Close'], 1)
    # X_train = train_df.drop(['Stock'], 1)

    y_valid = valid_df['Close'].copy()
    X_valid = valid_df.drop(['Close'], 1)
    # X_valid = valid_df.drop(['Stock'], 1)
    y_test  = test_df['Close'].copy()
    X_test  = test_df.drop(['Close'], 1)
    return X_train, y_train, X_test, y_test, X_valid, y_valid, indexs

def load_RNN(meta_file,x_test,y_test):
    tf.disable_v2_behavior() 
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(meta_file)
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    outputs2 = tf.get_collection('outputs')
    y_pred = sess.run(outputs2[0],feed_dict={outputs2[1]: x_test})
    rmse = np.sqrt(np.mean(y_pred - y_test)**2)
    print('rmse: ', rmse)
    return y_pred


def save_pickle(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))

def create_data(dataset, ratio, number_of_previous_close_prices=0):
    training_data_len = math.ceil(len(dataset)*ratio)
    return dataset[0:training_data_len,:], dataset[training_data_len - number_of_previous_close_prices:,:]


def create_x_y_train(data, number_of_previous_close_prices):
    x_train_data,y_train_data=[],[]
    for i in range(number_of_previous_close_prices,len(data)):
        x_train_data.append(data[i-number_of_previous_close_prices:i,0])
        y_train_data.append(data[i,0])
    
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

    return x_train_data, y_train_data


def create_x_y_test(data, number_of_previous_close_prices):
    print(len(data))
    x_test = []
    y_test = data[number_of_previous_close_prices:]
    for i in range(number_of_previous_close_prices,len(data)):
        x_test.append(data[i-number_of_previous_close_prices:i,0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return x_test, y_test

#save model
def save_model(model,filename):
    joblib.dump(model, filename)
#load model
def load_model (filename):
    loaded_model = joblib.load(filename)
    return loaded_model


# chạy các models được sử dụng với LogisticRegression
def Lr_model(x_train, y_train, x_test, y_test, parameters):
    lr=LogisticRegression(**parameters, n_jobs = -1)
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)
    print("LogisticRegression Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return metrics.accuracy_score(y_test, y_pred)

# chạy các models được sử dụng với RandomForestClassifier
def Rf_model(x_train, y_train, x_test, y_test, parameters):
    rfc=RandomForestClassifier(**parameters)
    rfc.fit(x_train,y_train)
    y_pred=rfc.predict(x_test)
    print("RandomForestClassifier Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return metrics.accuracy_score(y_test, y_pred)

#SVR
def SVR_model(x_train, y_train, x_test, y_test, parameters):
    svr=SVR(**parameters)
    svr.fit(x_train,y_train)
    y_pred=svr.predict(x_test)
    print("SVR Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # print("SVR Accuracy:",svr.score(y_test, y_pred))
    #get the root mean squared error(RMSE)
    rmse = np.sqrt(np.mean(y_pred - y_test)**2)
    print('rmse: ', rmse)
    return metrics.accuracy_score(y_test, y_pred)


# chạy các models được sử dụng với SVC
def Svc_model(x_train, y_train, x_test, y_test, parameters):
    svcc=SVC(**parameters)
    svcc.fit(x_train,y_train)
    y_pred=svcc.predict(x_test)
    print("SVC Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return metrics.accuracy_score(y_test, y_pred)

# chạy các models được sử dụng với KNeighborsClassifier
def Knf_model(x_train, y_train, x_test, y_test, parameters):
    knf=KNeighborsClassifier(**parameters)
    knf.fit(x_train,y_train)
    y_pred=knf.predict(x_test)
    print("KNeighborsClassifier Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return metrics.accuracy_score(y_test, y_pred)

# chạy các models được sử dụng với DecisionTreeClassifier
def Dtc_model(x_train, y_train, x_test, y_test, parameters):
    dtc=DecisionTreeClassifier(**parameters)
    dtc.fit(x_train,y_train)
    y_pred=dtc.predict(x_test)
    print("DecisionTreeClassifier Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return metrics.accuracy_score(y_test, y_pred)
# Function chạy tổng hợp các models tốt nhất của các thư viện sử dụng và chọn ra kết quả tốt nhất
def test_all(x_train, y_train, x_test, y_test, parameters):
    score_list=[]
    for m in parameters:
        name = m['name']
        params = m['params']
        if(name == 'LogisticRegression'):
            score_list.append(Lr_model(x_train, y_train, x_test, y_test, params))
        if(name == 'RandomForestClassifier'):
            score_list.append(Rf_model(x_train, y_train, x_test, y_test, params))
        if(name == 'SVC'):
            score_list.append(Svc_model(x_train, y_train, x_test, y_test, params))
        if(name == 'KNeighborsClassifier'):
            score_list.append(Knf_model(x_train, y_train, x_test, y_test, params))
        if(name == 'DecisionTreeClassifier'):
            score_list.append(Dtc_model(x_train, y_train, x_test, y_test, params))
    print('----------')
    #print(score_list)
    print("Best Accuracy: ",max(score_list))


def load_params():
    params = []
    params.append({'name':'LogisticRegression','params':{}})
    params.append({'name':'LogisticRegression','params':load_model('LF_params.txt').best_params_})
    params.append({'name':'RandomForestClassifier','params':{}})
    params.append({'name':'RandomForestClassifier','params':load_model('RandomForestClassifier.txt').best_params_})
    params.append({'name':'SVC','params':{}})
    params.append({'name':'SVC','params':load_model('SVC.txt').best_params_})
    params.append({'name':'KNeighborsClassifier','params':{}})
    params.append({'name':'KNeighborsClassifier','params':load_model('KNeighborsClassifier.txt').best_params_})
    params.append({'name':'DecisionTreeClassifier','params':{}})
    params.append({'name':'DecisionTreeClassifier','params':load_model('DecisionTreeClassifier.txt').best_params_})
    return params