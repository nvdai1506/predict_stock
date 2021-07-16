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
tf.disable_v2_behavior() 


def load_RNN(meta_file,x_test,y_test):   
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph(meta_file)
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    outputs2 = tf.get_collection('outputs')
    y_pred = sess.run(outputs2[0],feed_dict={outputs2[1]: x_test})
    rmse = np.sqrt(np.mean(y_pred - y_test)**2)
    print('rmse: ', rmse)


def save_pickle(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))

def create_data(dataset, ratio):
    training_data_len = math.ceil(len(dataset)*ratio)
    return dataset[0:training_data_len,:], dataset[training_data_len:,:]


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