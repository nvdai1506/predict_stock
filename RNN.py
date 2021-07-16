import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import utils as u
import math
import tensorflow as tf

print(tf.version.VERSION)

df = pd.read_csv("NSE-TATA.csv")
df.head()
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']
#Sort the dataset on date time and # filter "Date" and "Close" columns:
data = df.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
for i in range(0, len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["Close"][i] = data["Close"][i]

# Normalize the new filtered dataset:
new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)
dataset = new_dataset.values
dataset = dataset.flatten()
series = np.array(dataset)
n_windows = 10
n_input = 1
n_output = 1
size_train = 1001

## Split data
train = series[:size_train]
test = series[size_train:1232]
print(train.shape, test.shape)


def create_batches(windows, input, output):
    ## Create X
    x_data = train[:size_train-1]  # Select the data
    x_batches = x_data.reshape(-1, windows, input)  # Reshape the data
    x_test_data = test[:len(test)-1]
    x_test = x_test_data.reshape(-1, windows, input)  # Reshape the data

    ## Create y
    y_data = train[n_output:size_train]
    y_batches = y_data.reshape(-1, windows, output)
    y_test_data = test[n_output:]
    y_test = y_test_data.reshape(-1, windows, output)
    return x_batches, y_batches, x_test, y_test


x_batches, y_batches, x_test, y_test = create_batches(windows=n_windows,
                                                      input=n_input,
                                                      output=n_output)

print(test.shape)
print(x_batches.shape, y_batches.shape)
print(x_test.shape, y_test.shape)

tf.compat.v1.disable_eager_execution()
tf.compat.v1.placeholder(tf.float32, [None, n_windows, n_input])

tf.compat.v1.reset_default_graph()
r_neuron = 120

## 1. Construct the tensors
X = tf.compat.v1.placeholder(tf.float32, [None, n_windows, n_input])
y = tf.compat.v1.placeholder(tf.float32, [None, n_windows, n_output])

## 2. create the model
basic_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(
    num_units=r_neuron, activation=tf.nn.relu)
rnn_output, states = tf.compat.v1.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
stacked_rnn_output = tf.reshape(rnn_output, [-1, r_neuron])
stacked_outputs = tf.compat.v1.layers.dense(stacked_rnn_output, n_output)
outputs = tf.reshape(stacked_outputs, [-1, n_windows, n_output])

## 3. Loss + optimization
learning_rate = 0.001
loss = tf.reduce_sum(input_tensor=tf.square(outputs - y))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()
tf.compat.v1.add_to_collection('outputs', outputs)
iteration = 1500

with tf.compat.v1.Session() as sess:
    init.run()
    for iters in range(iteration):
        sess.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if iters % 150 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(iters, "\tMSE:", mse)
    print('outputs: ', outputs)
    y_pred = sess.run(outputs, feed_dict={X: x_test})
    save_path = saver.save(sess, "RNN.ckpt")
    print("Model saved in path: %s" % save_path)

y_test = y_test.flatten()
y_pred = y_pred.flatten()

#get the root mean squared error(RMSE)
rmse = np.sqrt(np.mean(y_pred - y_test)**2)
print('rmse: ', rmse)


def load_RNN(meta_file, x_test, y_test):
    sess = tf.compat.v1.Session()
    new_saver = tf.compat.v1.train.import_meta_graph(meta_file)
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    outputs2 = tf.compat.v1.get_collection('outputs')
    y_pred = sess.run(outputs2[0], feed_dict={X: x_test})
    rmse = np.sqrt(np.mean(y_pred - y_test)**2)
    print('rmse: ', rmse)


load_RNN("RNN.ckpt.meta", x_test, y_test)

plt.title("Forecast vs Actual", fontsize=14)
plt.plot(pd.Series(np.ravel(y_test)), "bo",
         markersize=8, label="Actual", color='green')
plt.plot(pd.Series(np.ravel(y_pred)), "r.",
         markersize=8, label="Forecast", color='red')
plt.legend(loc="lower left")
plt.xlabel("Time")
plt.show()
