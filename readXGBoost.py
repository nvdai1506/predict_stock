import utils as u
import xgboost as xgb
import pandas as pd
import numpy as np

df=pd.read_csv("NSE-TATA.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.head()

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
#MACD
EMA_12 = pd.Series(df['Close'].ewm(span=12, min_periods=12).mean())
EMA_26 = pd.Series(df['Close'].ewm(span=26, min_periods=26).mean())
df['MACD'] = pd.Series(EMA_12 - EMA_26)
df['MACD_signal'] = pd.Series(df.MACD.ewm(span=9, min_periods=9).mean())
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
X_test  = test_df.drop(['Close'], 1)

drop_cols = ['Date', 'Open', 'Low', 'High']
train_df = train_df.drop(drop_cols, 1)
valid_df = valid_df.drop(drop_cols, 1)

y_train = train_df['Close'].copy()
X_train = train_df.drop(['Close'], 1)
# X_train = train_df.drop(['Stock'], 1)
y_valid = valid_df['Close'].copy()
X_valid = valid_df.drop(['Close'], 1)

#load & test
eval_set = [(X_train, y_train), (X_valid, y_valid)]
clf2 = u.load_model('XGBoost.txt')
model2 = xgb.XGBRegressor(**clf2.best_params_, objective='reg:squarederror')
model2.fit(X_train, y_train, eval_set=eval_set, verbose=False)
## Calculate and visualize predictions
y_pred2 = model2.predict(X_test)
print(y_pred2)