#!/usr/bin/env python
# coding: utf-8

# <img src="https://certificate.tpq.io/taim_logo.png" width="350px" align="right">

# # Artificial Intelligence in Finance

# ## Risk Management

# Dr Yves J Hilpisch | The AI Machine
# 
# http://aimachine.io | http://twitter.com/dyjh

# In[1]:


import os
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


# ## Trading Bot

# In[2]:


import finance
import tradingbot


# In[3]:


symbol = 'EUR='
features = [symbol, 'r', 's', 'm', 'v']


# In[4]:


a = 0
b = 1750
c = 250


# In[5]:


learn_env = finance.Finance(symbol, features, window=20, lags=3,
                 leverage=1, min_performance=0.9, min_accuracy=0.475,
                 start=a, end=a + b, mu=None, std=None)


# In[6]:


learn_env.data.info()


# In[7]:


valid_env = finance.Finance(symbol, features=learn_env.features,
                            window=learn_env.window,
                            lags=learn_env.lags,
                            leverage=learn_env.leverage,
                            min_performance=0.0, min_accuracy=0.0,
                            start=a + b, end=a + b + c,
                            mu=learn_env.mu, std=learn_env.std)


# In[8]:


valid_env.data.info()


# In[9]:


tradingbot.set_seeds(100)
agent = tradingbot.TradingBot(24, 0.001, learn_env, valid_env)


# In[10]:


episodes = 61


# In[11]:


get_ipython().run_line_magic('time', 'agent.learn(episodes)')


# In[12]:


tradingbot.plot_treward(agent)


# In[13]:


tradingbot.plot_performance(agent)


# ## Vectorized Backtesting

# In[14]:


def reshape(s):
    return np.reshape(s, [1, learn_env.lags,
                          learn_env.n_features])


# In[15]:


def backtest(agent, env):
    env.min_accuracy = 0.0
    env.min_performance = 0.0
    done = False
    env.data['p'] = 0
    state = env.reset()
    while not done:
        action = np.argmax(
            agent.model.predict(reshape(state))[0, 0])
        position = 1 if action == 1 else -1
        env.data.loc[:, 'p'].iloc[env.bar] = position
        state, reward, done, info = env.step(action)
    env.data['s'] = env.data['p'] * env.data['r'] * learn_env.leverage


# In[16]:


env = agent.learn_env


# In[17]:


backtest(agent, env)


# In[18]:


env.data['p'].iloc[env.lags:].value_counts()


# In[19]:


env.data[['r', 's']].iloc[env.lags:].sum().apply(np.exp)


# In[20]:


env.data[['r', 's']].iloc[env.lags:].sum().apply(np.exp) - 1


# In[21]:


env.data[['r', 's']].iloc[env.lags:].cumsum(
        ).apply(np.exp).plot(figsize=(10, 6));


# In[22]:


test_env = finance.Finance(symbol, features=learn_env.features,
                           window=learn_env.window,
                           lags=learn_env.lags,
                           leverage=learn_env.leverage,
                           min_performance=0.0, min_accuracy=0.0,
                           start=a + b + c, end=None,
                           mu=learn_env.mu, std=learn_env.std)


# In[23]:


env = test_env


# In[24]:


backtest(agent, env)


# In[25]:


env.data['p'].iloc[env.lags:].value_counts()


# In[26]:


env.data[['r', 's']].iloc[env.lags:].sum().apply(np.exp)


# In[27]:


env.data[['r', 's']].iloc[env.lags:].sum().apply(np.exp) - 1


# In[28]:


env.data[['r', 's']].iloc[env.lags:].cumsum(
            ).apply(np.exp).plot(figsize=(10, 6));


# ## Event-Based Backtesting

# In[29]:


import backtesting as bt


# In[30]:


bb = bt.BacktestingBase(env=agent.learn_env, model=agent.model,
                        amount=10000, ptc=0.0001, ftc=1.0,
                        verbose=True) 


# In[31]:


bb.initial_amount


# In[32]:


bar = 100


# In[33]:


bb.get_date_price(bar)


# In[34]:


bb.env.get_state(bar)


# In[35]:


bb.place_buy_order(bar, amount=5000)


# In[36]:


bb.print_net_wealth(2 * bar)


# In[37]:


bb.place_sell_order(2 * bar, units=1000)


# In[38]:


bb.close_out(3 * bar)


# In[39]:


class TBBacktester(bt.BacktestingBase):
    def _reshape(self, state):
        ''' Helper method to reshape state objects.
        '''
        return np.reshape(state, [1, self.env.lags, self.env.n_features])
    def backtest_strategy(self):
        ''' Event-based backtesting of the trading bot's performance.
        '''
        self.units = 0
        self.position = 0
        self.trades = 0
        self.current_balance = self.initial_amount
        self.net_wealths = list()
        for bar in range(self.env.lags, len(self.env.data)):
            date, price = self.get_date_price(bar)
            if self.trades == 0:
                print(50 * '=')
                print(f'{date} | *** START BACKTEST ***')
                self.print_balance(bar)
                print(50 * '=')
            state = self.env.get_state(bar)
            action = np.argmax(self.model.predict(
                        self._reshape(state.values))[0, 0])
            position = 1 if action == 1 else -1
            if self.position in [0, -1] and position == 1:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOING LONG ***')
                if self.position == -1:
                    self.place_buy_order(bar - 1, units=-self.units)
                self.place_buy_order(bar - 1,
                                     amount=self.current_balance)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = 1
            elif self.position in [0, 1] and position == -1:
                if self.verbose:
                    print(50 * '-')
                    print(f'{date} | *** GOING SHORT ***')
                if self.position == 1:
                    self.place_sell_order(bar - 1, units=self.units)
                self.place_sell_order(bar - 1,
                                      amount=self.current_balance)
                if self.verbose:
                    self.print_net_wealth(bar)
                self.position = -1
            self.net_wealths.append((date,
                                     self.calculate_net_wealth(price)))
        self.net_wealths = pd.DataFrame(self.net_wealths,
                                        columns=['date', 'net_wealth'])
        self.net_wealths.set_index('date', inplace=True)
        self.net_wealths.index = pd.DatetimeIndex(
                                        self.net_wealths.index)
        self.close_out(bar)


# In[40]:


env = learn_env


# In[41]:


tb = TBBacktester(env, agent.model, 10000,
                  0.0, 0, verbose=False)


# In[42]:


tb.backtest_strategy()


# In[43]:


tb_ = TBBacktester(env, agent.model, 10000,
                   0.00012, 0.0, verbose=False)


# In[44]:


tb_.backtest_strategy()


# In[45]:


ax = tb.net_wealths.plot(figsize=(10, 6))
tb_.net_wealths.columns = ['net_wealth (after tc)']
tb_.net_wealths.plot(ax=ax);


# In[46]:


env = test_env


# In[47]:


tb = TBBacktester(env, agent.model, 10000,
                  0.0, 0, verbose=False)


# In[48]:


tb.backtest_strategy()


# In[49]:


tb_ = TBBacktester(env, agent.model, 10000,
                   0.00012, 0.0, verbose=False)


# In[50]:


tb_.backtest_strategy()


# In[51]:


ax = tb.net_wealths.plot(figsize=(10, 6))
tb_.net_wealths.columns = ['net_wealth (after tc)']
tb_.net_wealths.plot(ax=ax);


# In[52]:


ax = (tb.net_wealths / tb.net_wealths.iloc[0]).plot(figsize=(10, 6))
tp = env.data[['r', 's']].iloc[env.lags:].cumsum().apply(np.exp)
(tp / tp.iloc[0]).plot(ax=ax);


# ## Assessing Risk

# In[53]:


data = pd.DataFrame(learn_env.data[symbol])


# In[54]:


data.head()


# In[55]:


window = 14


# In[56]:


data['min'] = data[symbol].rolling(window).min()


# In[57]:


data['max'] = data[symbol].rolling(window).max()


# In[58]:


data['mami'] = data['max'] - data['min']


# In[59]:


data['mac'] = abs(data['max'] - data[symbol].shift(1))


# In[60]:


data['mic'] = abs(data['min'] - data[symbol].shift(1))


# In[61]:


data['atr'] = np.maximum(data['mami'], data['mac'])


# In[62]:


data['atr'] = np.maximum(data['atr'], data['mic'])


# In[63]:


data['atr%'] = data['atr'] / data[symbol]


# In[64]:


data[['atr', 'atr%']].plot(subplots=True, figsize=(10, 6));


# In[65]:


data[['atr', 'atr%']].tail()


# In[66]:


leverage = 10


# In[67]:


data[['atr', 'atr%']].tail() * leverage


# In[68]:


data[['atr', 'atr%']].median() * leverage


# ## Backtesting Risk Measures

# In[69]:


import tbbacktesterrm as tbbrm


# In[70]:


env = test_env


# In[71]:


tb = tbbrm.TBBacktesterRM(env, agent.model, 10000,
                          0.0, 0, verbose=False)


# In[72]:


tb.backtest_strategy(sl=None, tsl=None, tp=None, wait=5)


# ### Stop Loss

# In[73]:


tb.backtest_strategy(sl=0.0175, tsl=None, tp=None,
                     wait=5, guarantee=False)


# In[74]:


tb.backtest_strategy(sl=0.017, tsl=None, tp=None,
                     wait=5, guarantee=True)


# ### Trailing Stop Loss

# In[75]:


tb.backtest_strategy(sl=None, tsl=0.015,
                     tp=None, wait=5)


# ### Take Profit

# In[76]:


tb.backtest_strategy(sl=None, tsl=None, tp=0.015,
                     wait=5, guarantee=False)


# In[77]:


tb.backtest_strategy(sl=None, tsl=None, tp=0.015,
                     wait=5, guarantee=True)


# ## Combinations

# In[78]:


tb.backtest_strategy(sl=0.015, tsl=None,
                     tp=0.0185, wait=5)


# In[79]:


tb.backtest_strategy(sl=None, tsl=0.02,
                     tp=0.02, wait=5)


# <img src='http://hilpisch.com/taim_logo.png' width="350px" align="right">
# 
# <br><br><br><a href="http://tpq.io" target="_blank">http://tpq.io</a> | <a href="http://twitter.com/dyjh" target="_blank">@dyjh</a> | <a href="mailto:ai@tpq.io">ai@tpq.io</a>
