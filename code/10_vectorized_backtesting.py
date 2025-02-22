#!/usr/bin/env python
# coding: utf-8

# <img src="https://certificate.tpq.io/taim_logo.png" width="350px" align="right">

# # Artificial Intelligence in Finance

# ## Vectorized Backtesting

# Dr Yves J Hilpisch | The AI Machine
# 
# http://aimachine.io | http://twitter.com/dyjh

# In[1]:


import os
import math
import numpy as np
import pandas as pd
from pylab import plt, mpl
plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.float_format', '{:.4f}'.format)
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'


# ## Backtesting an SMA-Based Strategy

# In[2]:


url = 'http://hilpisch.com/aiif_eikon_eod_data.csv'


# In[3]:


symbol = 'EUR='


# In[4]:


data = pd.DataFrame(pd.read_csv(url, index_col=0,
                                parse_dates=True).dropna()[symbol])


# In[5]:


data.info()


# In[6]:


data['SMA1'] = data[symbol].rolling(42).mean()


# In[7]:


data['SMA2'] = data[symbol].rolling(258).mean()


# In[8]:


data.plot(figsize=(10, 6));


# In[9]:


data.dropna(inplace=True)


# In[10]:


data['p'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)


# In[11]:


data['p'] = data['p'].shift(1)


# In[12]:


data.dropna(inplace=True)


# In[13]:


data.plot(figsize=(10, 6), secondary_y='p');


# In[14]:


data['r'] = np.log(data[symbol] / data[symbol].shift(1))


# In[15]:


data.dropna(inplace=True)


# In[16]:


data['s'] = data['p'] * data['r']


# In[17]:


data[['r', 's']].sum().apply(np.exp)  # gross performance


# In[18]:


data[['r', 's']].sum().apply(np.exp) - 1  # net performance


# In[19]:


data[['r', 's']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# In[20]:


sum(data['p'].diff() != 0) + 1


# In[21]:


pc = 0.005


# In[22]:


data['s_'] = np.where(data['p'].diff() != 0,
                      data['s'] - pc, data['s'])


# In[23]:


# data['s_'].iloc[0] -= pc


# In[24]:


data['s_'].iloc[-1] -= pc


# In[25]:


data[['r', 's', 's_']][data['p'].diff() != 0]


# In[26]:


data[['r', 's', 's_']].sum().apply(np.exp)


# In[27]:


data[['r', 's', 's_']].sum().apply(np.exp) - 1


# In[28]:


data[['r', 's', 's_']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# In[29]:


data[['r', 's', 's_']].std()


# In[30]:


data[['r', 's', 's_']].std() * math.sqrt(252)


# ## Backtesting a Daily DNN-Based Strategy

# In[31]:


data = pd.DataFrame(pd.read_csv(url, index_col=0,
                                parse_dates=True).dropna()[symbol])


# In[32]:


data.info()


# In[33]:


lags = 5


# In[34]:


def add_lags(data, symbol, lags, window=20):
    cols = []
    df = data.copy()
    df.dropna(inplace=True)
    df['r'] = np.log(df / df.shift(1))
    df['sma'] = df[symbol].rolling(window).mean()
    df['min'] = df[symbol].rolling(window).min()
    df['max'] = df[symbol].rolling(window).max()
    df['mom'] = df['r'].rolling(window).mean()
    df['vol'] = df['r'].rolling(window).std()
    df.dropna(inplace=True)
    df['d'] = np.where(df['r'] > 0, 1, 0)
    features = [symbol, 'r', 'd', 'sma', 'min', 'max', 'mom', 'vol']
    for f in features:
        for lag in range(1, lags + 1):
            col = f'{f}_lag_{lag}'
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols


# In[35]:


data, cols = add_lags(data, symbol, lags, window=20)


# In[36]:


import random
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.metrics import accuracy_score


# In[37]:


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seeds()


# In[38]:


optimizer = keras.optimizers.legacy.Adam(learning_rate=0.0001)


# In[39]:


def create_model(hl=2, hu=128, dropout=False, rate=0.3,
                regularize=False, reg=l1(0.0005),
                optimizer=optimizer, input_dim=len(cols)):
    if not regularize:
        reg = None
    model = Sequential()
    model.add(Dense(hu, input_dim=input_dim,
                 activity_regularizer=reg,  
                 activation='relu'))
    if dropout:
        model.add(Dropout(rate, seed=100))
    for _ in range(hl):
        model.add(Dense(hu, activation='relu',
                     activity_regularizer=reg))
        if dropout:
            model.add(Dropout(rate, seed=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


# In[40]:


split = '2018-01-01'


# In[41]:


train = data.loc[:split].copy()


# In[42]:


np.bincount(train['d'])


# In[43]:


mu, std = train.mean(), train.std()


# In[44]:


train_ = (train - mu) / std


# In[45]:


set_seeds()
model = create_model(hl=2, hu=64)


# In[46]:


get_ipython().run_cell_magic('time', '', "model.fit(train_[cols], train['d'],\n        epochs=20, verbose=False,\n        validation_split=0.2, shuffle=False)")


# In[47]:


model.evaluate(train_[cols], train['d'])


# In[48]:


train['p'] = np.where(model.predict(train_[cols]) > 0.5, 1, 0)


# In[49]:


train['p'] = np.where(train['p'] == 1, 1, -1)


# In[50]:


train['p'].value_counts()


# In[51]:


train['s'] = train['p'] * train['r']


# In[52]:


train[['r', 's']].sum().apply(np.exp)


# In[53]:


train[['r', 's']].sum().apply(np.exp)  - 1


# In[54]:


train[['r', 's']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# In[55]:


test = data.loc[split:].copy()


# In[56]:


test_ = (test - mu) / std


# In[57]:


model.evaluate(test_[cols], test['d'])


# In[58]:


test['p'] = np.where(model.predict(test_[cols]) > 0.5, 1, -1)


# In[59]:


test['p'].value_counts()


# In[60]:


test['s'] = test['p'] * test['r']


# In[61]:


test[['r', 's']].sum().apply(np.exp)


# In[62]:


test[['r', 's']].sum().apply(np.exp) - 1


# In[63]:


test[['r', 's']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# In[64]:


sum(test['p'].diff() != 0) + 1


# In[65]:


spread = 0.00012
pc = spread / data[symbol].mean()
print(f'{pc:.6f}')


# In[66]:


test['s_'] = np.where(test['p'].diff() != 0,
                      test['s'] - pc, test['s'])


# In[67]:


# test['s_'].iloc[0] -= pc


# In[68]:


test['s_'].iloc[-1] -= pc


# In[69]:


test[['r', 's', 's_']].sum().apply(np.exp)


# In[70]:


test[['r', 's', 's_']].sum().apply(np.exp) - 1


# In[71]:


test[['r', 's', 's_']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# ## Backtesting an Intraday DNN-Based Strategy

# In[72]:


url = 'http://hilpisch.com/aiif_eikon_id_eur_usd.csv'


# In[73]:


symbol = 'EUR='


# In[74]:


data = pd.DataFrame(pd.read_csv(url, index_col=0,
                    parse_dates=True).dropna()['CLOSE'])
data.columns = [symbol]


# In[75]:


data = data.resample('5min', label='right').last().ffill()


# In[76]:


data.info()


# In[77]:


data.head()


# In[78]:


data[symbol].plot(figsize=(10, 6));


# In[79]:


lags = 5


# In[80]:


data, cols = add_lags(data, symbol, lags, window=20)


# In[81]:


split = int(len(data) * 0.85)


# In[82]:


train = data.iloc[:split].copy()


# In[83]:


np.bincount(train['d'])


# In[84]:


def cw(df):
    c0, c1 = np.bincount(df['d'])
    w0 = (1 / c0) * (len(df)) / 2
    w1 = (1 / c1) * (len(df)) / 2
    return {0: w0, 1: w1}


# In[85]:


mu, std = train.mean(), train.std()


# In[86]:


train_ = (train - mu) / std


# In[87]:


set_seeds()
model = create_model(hl=1, hu=128,
                     reg=True, dropout=False)


# In[88]:


get_ipython().run_cell_magic('time', '', "model.fit(train_[cols], train['d'],\n          epochs=40, verbose=False,\n          validation_split=0.2, shuffle=False,\n          class_weight=cw(train))")


# In[89]:


model.evaluate(train_[cols], train['d'])


# In[90]:


train['p'] = np.where(model.predict(train_[cols]) > 0.5, 1, -1)


# In[91]:


train['p'].value_counts()


# In[92]:


train['s'] = train['p'] * train['r']


# In[93]:


train[['r', 's']].sum().apply(np.exp)


# In[94]:


train[['r', 's']].sum().apply(np.exp) - 1


# In[95]:


train[['r', 's']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# In[96]:


test = data.iloc[split:].copy()


# In[97]:


test_ = (test - mu) / std


# In[98]:


model.evaluate(test_[cols], test['d'])


# In[99]:


test['p'] = np.where(model.predict(test_[cols]) > 0.5, 1, -1)


# In[100]:


test['p'].value_counts()


# In[101]:


test['s'] = test['p'] * test['r']


# In[102]:


test[['r', 's']].sum().apply(np.exp)


# In[103]:


test[['r', 's']].sum().apply(np.exp) - 1


# In[104]:


test[['r', 's']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# In[105]:


sum(test['p'].diff() != 0) + 1


# In[106]:


spread = 0.00012
pc_1 = spread / test[symbol]


# In[107]:


spread = 0.00006
pc_2 = spread / test[symbol]


# In[108]:


test['s_1'] = np.where(test['p'].diff() != 0,
                       test['s'] - pc_1, test['s'])


# In[109]:


# test['s_1'].iloc[0] -= pc_1.iloc[0]
test['s_1'].iloc[-1] -= pc_1.iloc[0]


# In[110]:


test['s_2'] = np.where(test['p'].diff() != 0,
                       test['s'] - pc_2, test['s'])


# In[111]:


# test['s_2'].iloc[0] -= pc_2.iloc[0]
test['s_2'].iloc[-1] -= pc_2.iloc[0]


# In[112]:


test[['r', 's', 's_1', 's_2']].sum().apply(np.exp)


# In[113]:


test[['r', 's', 's_1', 's_2']].sum().apply(np.exp) - 1


# In[114]:


test[['r', 's', 's_1', 's_2']].cumsum().apply(
    np.exp).plot(figsize=(10, 6), style=['-', '-', '--', '--']);


# <img src='http://hilpisch.com/taim_logo.png' width="350px" align="right">
# 
# <br><br><br><a href="http://tpq.io" target="_blank">http://tpq.io</a> | <a href="http://twitter.com/dyjh" target="_blank">@dyjh</a> | <a href="mailto:ai@tpq.io">ai@tpq.io</a>
