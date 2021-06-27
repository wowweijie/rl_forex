import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench.monitor import Monitor
from stable_baselines import logger

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                dfs_list,
                hmax,                
                initial_amount,
                buy_cost_pct,
                sell_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                turbulence_threshold=None,
                make_plots = False, 
                print_verbosity = 10,
                timestep = 0, 
                initial=True,
                previous_state=[],
                model_name = '',
                mode='',
                iteration=''):
        self.timestep = timestep

        #concat dfs into one
        df = pd.concat(list(dfs_list.values()), axis = 1, keys = list(dfs_list.keys()))
        df.reset_index(inplace = True, col_fill='date')

        #differentiate USD quoted currency as True, and the rest as False
        self.usd_quoted = list(map(lambda symbol: symbol[-3:]=="USD", dfs_list.keys()))
        self.ccy_list = list(dfs_list.keys())
        self.df = df
        self.stock_dim = len(self.ccy_list) 
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.action_space,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space,))
        self.data = self.df.iloc[self.timestep,:]
        self.terminal = False     
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.model_name=model_name
        self.mode=mode 
        self.iteration=iteration
        # initalize state
        # usd balance, close price, positions in other ccy, technical indicators
        self.state = self._init_state()
        
        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        #self.reset()
        self._seed()

        #initialize episode reward mem
        self.episode_rewards = []

    def _calculate_nop(self) -> float:
        return sum(
                    np.array(self.state[1:self.stock_dim+1]) *\
                    np.array(self.state[self.stock_dim*2+1:self.stock_dim*3+1]) *\
                    np.array(list(map(lambda x : 1 if x else 0, self.usd_quoted)))
                ) + sum(
                    np.array(self.state[self.stock_dim*2+1:self.stock_dim*3+1]) /\
                    np.array(self.state[1:self.stock_dim+1]) *\
                    np.array(list(map(lambda x : 0 if x else 1, self.usd_quoted)))
                ) + self.state[0]
        


    def _sell_stock(self, index, action):

        # For USD-quoted currencies, i.e. GBPUSD, EURUSD
        if self.usd_quoted[index]:
            base_ccy_index = index+2*self.stock_dim+1
            quote_ccy_index = 0

        # For USD-based currencies, i.e. USDCHF, USDJPY
        else:  
            base_ccy_index = 0
            quote_ccy_index = index+2*self.stock_dim+1

        close_price_index = index+1
        close_price = self.state[close_price_index]

        # Sell only if the price is > 0 (no missing data in this particular date)
        # Sell only if current asset is > 0
        if self.state[index+1]>0 and self.state[base_ccy_index] > 0:
                
            sell_num_lots = min(abs(action),self.state[base_ccy_index])
            sell_amount = close_price * sell_num_lots
            cost = sell_amount * self.sell_cost_pct
            self.cost += cost
            
            # increase USD balance for USD-quoted currencies, i.e. GBPUSD, EURUSD
            # increase non-USD balance for USD-based currencies, i.e. USDCHF, USDJPY
            self.state[quote_ccy_index] += sell_amount - cost

            # deduct USD balance for USD-quoted currencies, i.e. GBPUSD, EURUSD
            # deduct non-USD balance for USD-based currencies, i.e. USDCHF, USDJPY
            self.state[base_ccy_index] -= sell_num_lots
            
            self.trades+=1
        else:
            sell_num_lots = 0


        return sell_num_lots

    
    def _buy_stock(self, index, action):

        # For USD-quoted currencies, i.e. GBPUSD, EURUSD
        if self.usd_quoted[index]:
            base_ccy_index = index+2*self.stock_dim+1
            quote_ccy_index = 0

        # For USD-based currencies, i.e. USDCHF, USDJPY
        else:  
            base_ccy_index = 0
            quote_ccy_index = index+2*self.stock_dim+1

        close_price_index = self.stock_dim+index+1
        close_price = self.state[close_price_index]

        # Buy only if the price is > 0 (no missing data in this particular date)
        # Buy only if current asset is > 0
        if self.state[index+1]>0 and self.state[quote_ccy_index]>0: 

            # Quote currency balance available for buy order       
            available_amount = self.state[quote_ccy_index] // close_price
            
            # Update balance
            buy_num_lots = min(available_amount, action)
            buy_amount = close_price * buy_num_lots * (1+ self.buy_cost_pct)
            cost = buy_amount * self.buy_cost_pct
            self.cost += cost

            # deduct USD balance for USD-quoted currencies, i.e. GBPUSD, EURUSD
            # deduct non-USD balance for USD-based currencies, i.e. USDCHF, USDJPY
            self.state[quote_ccy_index] -= buy_amount + self.cost

            # increase USD balance for USD-quoted currencies, i.e. GBPUSD, EURUSD
            # increase non-USD balance for USD-based currencies, i.e. USDCHF, USDJPY
            self.state[base_ccy_index] += buy_num_lots

            self.trades+=1
        else:
            buy_num_lots = 0

        return buy_num_lots

    def step(self, actions):
        self.terminal = self.timestep >= len(self.df.index.unique())-1
        if self.terminal:        
            end_total_asset = self._calculate_nop()
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = end_total_asset - self.initial_amount 
            self.episode_rewards.append(tot_reward)
            if self.episode % self.print_verbosity == 0:
                print(f"step: {self.timestep}, episode: {self.episode}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                print("=================================")

            # Add outputs to logger interface
            logger.log("environment/nop_value", end_total_asset)
            logger.log("environment/total_reward", tot_reward)
            logger.log("environment/total_reward_pct", (tot_reward / (end_total_asset - tot_reward)) * 100)
            logger.log("environment/total_cost", self.cost)
            logger.log("environment/total_trades", self.trades)
            logger.log("train/episode_reward", self.reward)

            return self.state, self.reward, self.terminal, {}

        else:

            actions = actions * self.hmax #actions initially is scaled between 0 to 1
            actions = (actions.astype(int)) #convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence>=self.turbulence_threshold:
                    actions=np.array([-self.hmax]*self.stock_dim)
            begin_total_asset = self._calculate_nop()
            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            self.timestep += 1
            self.data = self.df.iloc[self.timestep,:]    
            self.state =  self._update_state()
                           
            end_total_asset = self._calculate_nop()
            self.reward = end_total_asset - begin_total_asset            
            self.reward = self.reward*self.reward_scaling

        return self.state, self.reward, self.terminal, {}

    def reset(self):  
        #initiate state
        self.state = self._init_state()
        
        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory = [previous_total_asset]

        self.timestep = 0
        self.data = self.df.iloc[self.timestep,:]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory=[]
        self.date_memory=[self._get_date()]
        
        self.episode+=1

        return self.state
    
    def render(self, mode='human',close=False):
        return self.state

    def _init_state(self):
        if self.initial:
            # For Initial State
            #if len(self.df.columns.levels[0])>0:
                # for multiple stock
                state = [self.initial_amount] + \
                         self.data[slice(None), "bid_close"].values.tolist() + \
                         self.data[slice(None), "ask_close"].values.tolist() + \
                         [0]*self.stock_dim  + \
                         sum([self.data[ccy][self.tech_indicator_list].values.tolist() for ccy in self.ccy_list] , [])
                         
            # else:
            #     # for single stock
            #     state = [self.initial_amount] + \
            #             [self.data.close] + \
            #             [0]*self.stock_dim  + \
            #             sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])

        else:
            #Using Previous State
            # if len(self.df.tic.unique())>1:
                # for multiple stock
                state = [self.previous_state[0]] + \
                         self.data[slice(None), "bid_close"].values.tolist() + \
                         self.data[slice(None), "ask_close"].values.tolist() + \
                         self.previous_state[(self.stock_dim*2+1):(self.stock_dim*3+1)]  + \
                         sum([self.data[ccy][self.tech_indicator_list].values.tolist() for ccy in self.ccy_list] , [])
            # else:
            #     # for single stock
            #     state = [self.previous_state[0]] + \
            #             [self.data.close] + \
            #             self.previous_state[(self.stock_dim+1):(self.stock_dim*2+1)]  + \
            #             sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
        return state

    def _update_state(self):
        #if len(self.df.tic.unique())>0:
            # for multiple stock
            state =  [self.state[0]] + \
                      self.data[slice(None), "bid_close"].values.tolist() + \
                      self.data[slice(None), "ask_close"].values.tolist() + \
                      list(self.state[(2*self.stock_dim+1):(self.stock_dim*3+1)]) + \
                      sum([self.data[ccy][self.tech_indicator_list].values.tolist() for ccy in self.ccy_list] , [])

        #else:
            # for single stock
            # state =  [self.state[0]] + \
            #          [self.data.close] + \
            #          list(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]) + \
            #          sum([[self.data[tech]] for tech in self.tech_indicator_list ], [])
                          
            return state

    def _get_date(self):
        #if len(self.df.tic.unique())>1:
            date = self.data["index", "date"]
        # else:
        #     date = self.data.date
            return date

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        #print(len(date_list))
        #print(len(asset_list))
        df_account_value = pd.DataFrame({'date':date_list,'account_value':asset_list})
        return df_account_value

    def save_action_memory(self):
        # if len(self.df.tic.unique())>1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.ccy_list
            df_actions.index = df_date.date
        # else:
        #     date_list = self.date_memory[:-1]
        #     action_list = self.actions_memory
        #     df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
            return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs