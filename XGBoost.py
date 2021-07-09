import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV

## Time series decomposition
from stldecompose import decompose

## Chart drawing
import plotly as py
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

## Mute sklearn warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

import joblib
##save model
def save_model(model,filename):
    joblib.dump(model, filename)
##load model
def load_model (filename):
    loaded_model = joblib.load(filename)
    return loaded_model

## Show charts when running kernel
init_notebook_mode(connected=True)

## Change default background color for all visualizations
layout=go.Layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(250,250,250,0.8)')
fig = go.Figure(layout=layout)
templated_fig = pio.to_templated(fig)
pio.templates['my_template'] = templated_fig.layout.template
pio.templates.default = 'my_template'

## Read the dataset:
df=pd.read_csv("NSE-TATA.csv")
df['Date'] = pd.to_datetime(df['Date'])
# df.head()

# fig = make_subplots(rows=2, cols=1)

# fig.add_trace(go.Ohlc(x=df.Date,
#                       open=df.Open,
#                       high=df.High,
#                       low=df.Low,
#                       close=df.Close,
#                       name='Price'), row=1, col=1)

# fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close'), row=2, col=1)

# fig.update(layout_xaxis_rangeslider_visible=False)
# fig.show()

##Decomposition
df_close = df[['Date', 'Close']].copy()
df_close = df_close.set_index('Date')
df_close.head()

decomp = decompose(df_close, period=365)
# fig = decomp.plot()
# fig.set_size_inches(20, 8)

# Technical indicators

##Moving Averages
df['EMA_9'] = df['Close'].ewm(9).mean().shift()
df['SMA_5'] = df['Close'].rolling(5).mean().shift()
df['SMA_10'] = df['Close'].rolling(10).mean().shift()
df['SMA_15'] = df['Close'].rolling(15).mean().shift()
df['SMA_30'] = df['Close'].rolling(30).mean().shift()

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df.Date, y=df.EMA_9, name='EMA 9'))
# fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_5, name='SMA 5'))
# fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_10, name='SMA 10'))
# fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_15, name='SMA 15'))
# fig.add_trace(go.Scatter(x=df.Date, y=df.SMA_30, name='SMA 30'))
# fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close', opacity=0.2))
# fig.show()

##Relative Strength Index
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

df['RSI'] = relative_strength_idx(df).fillna(0)

# fig = go.Figure(go.Scatter(x=df.Date, y=df.RSI, name='RSI'))
# fig.show()

##MACD
EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
df['MACD'] = pd.Series(EMA_12 - EMA_26)
df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())

# fig = make_subplots(rows=2, cols=1)
# fig.add_trace(go.Scatter(x=df.Date, y=df.Close, name='Close'), row=1, col=1)
# fig.add_trace(go.Scatter(x=df.Date, y=EMA_12, name='EMA 12'), row=1, col=1)
# fig.add_trace(go.Scatter(x=df.Date, y=EMA_26, name='EMA 26'), row=1, col=1)
# fig.add_trace(go.Scatter(x=df.Date, y=df['MACD'], name='MACD'), row=2, col=1)
# fig.add_trace(go.Scatter(x=df.Date, y=df['MACD_signal'], name='Signal line'), row=2, col=1)
# fig.show()

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

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=train_df.Date, y=train_df.Close, name='Training'))
# fig.add_trace(go.Scatter(x=valid_df.Date, y=valid_df.Close, name='Validation'))
# fig.add_trace(go.Scatter(x=test_df.Date,  y=test_df.Close,  name='Test'))
# fig.show()

## Drop unnecessary columns
drop_cols = ['Date', 'Open', 'Low', 'High']

train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)
test_df  = test_df.drop(drop_cols, 1)

## Split into features and labels
y_train = train_df['Close'].copy()
X_train = train_df.drop(['Close'], 1)
## X_train = train_df.drop(['Stock'], 1)

y_valid = valid_df['Close'].copy()
X_valid = valid_df.drop(['Close'], 1)
## X_valid = valid_df.drop(['Stock'], 1)

y_test  = test_df['Close'].copy()
X_test  = test_df.drop(['Close'], 1)

# X_train.info()


parameters = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'max_depth': [8, 10, 12, 15],
    'gamma': [0.001, 0.005, 0.01, 0.02],
    'random_state': [42]
}

eval_set = [(X_train, y_train), (X_valid, y_valid)]
model = xgb.XGBRegressor(eval_set=eval_set, objective='reg:squarederror', verbose=False)
clf = GridSearchCV(model, parameters)

clf.fit(X_train, y_train)

print(f'Best params: {clf.best_params_}')
print(f'Best validation score = {clf.best_score_}')



model = xgb.XGBRegressor(**clf.best_params_, objective='reg:squarederror')
model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# plot_importance(model)

## Calculate and visualize predictions
y_pred = model.predict(X_test)
print(f'y_true = {np.array(y_test)[:5]}')
print(f'y_pred = {y_pred[:5]}')

print(f'mean_squared_error = {mean_squared_error(y_test, y_pred)}')

## save model
save_model(clf,'XGBoost.txt')