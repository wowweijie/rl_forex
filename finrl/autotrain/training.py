import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import os
import json
from statistics import mean
from itertools import accumulate
from empyrical import sortino_ratio
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from random import sample

from finrl.config import config
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split, load_ohlc_dataset
from finrl.trade.backtest import evaluate_policy_rewards, evaluate_lstm_rewards
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.a2c import A2C
from finrl.trade.backtest import BackTestStats, BaselineStats, BackTestPlot

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import BackTestStats

def continual_training():
    train_iteration(1,17)
    train_iteration(2,17)
    train_iteration(3,17)
    train_iteration(4,17)

def create_env_kwargs(monthdata: str):
    EURUSD_df=load_ohlc_dataset(f"15min/EURUSD/{monthdata}.csv")
    GBPUSD_df=load_ohlc_dataset(f"15min/GBPUSD/{monthdata}.csv")
    USDJPY_df=load_ohlc_dataset(f"15min/USDJPY/{monthdata}.csv")
    USDCHF_df=load_ohlc_dataset(f"15min/USDJPY/{monthdata}.csv")


    param_map = {
                    'sma_9': {'talib_name' : 'SMA', 'time_period' : 9}, #time_period in seoonds
                    'ema_9': {'talib_name' : 'EMA', 'time_period' : 9}, #time_period in seoonds
                    'sma_21' : {'talib_name' : 'SMA', 'time_period' : 21},
                    'ema_21' : {'talib_name' : 'EMA', 'time_period' : 21},
                    'bbands_9':{'talib_name':'BBANDS','time_period':9,'nbdevup':2.0,'nbdevdn':2.0},
                    'bbands_12':{'talib_name':'BBANDS','time_period':12,'nbdevup':2.0,'nbdevdn':2.0},
                    'macd_entry':{'talib_name':'MACD', 'fastperiod':12, 'slowperiod':26,'signalperiod':9},
                    'macd_exit':{'talib_name':'MACD', 'fastperiod':19, 'slowperiod':39,'signalperiod':9},
                    'stoch':{'talib_name':'STOCH', 'fastk_period':5, 'slowk_period':3, 'slowk_matype':0, 'slowd_period':3, 'slowd_matype':0},
                    'rsi_14':{'talib_name':'RSI', 'time_period':14},
                    'rsi_4':{'talib_name':'RSI','time_period':4},
                    'mom_10':{'talib_name':'MOM', 'time_period':10},
                    'stochrsi_14':{'talib_name':'STOCHRSI', 'time_period':14, 'fastk_period':5,'fastd_period':3, 'fastd_matype':0},
                    'kama_30':{'talib_name':'KAMA', 'time_period':30},
                    't3_5':{'talib_name':'T3', 'time_period':5, 'vfactor':0.7},
                    'atr_14':{'talib_name':'ATR', 'time_period':14},
                    'natr_14':{'talib_name':'NATR', 'time_period':14},
                    'tsf_14':{'talib_name':'TSF', 'time_period':14},
    }

    EURUSD_train, tech_indicator_list = FeatureEngineer(EURUSD_df,
                            tech_indicator_params_map = param_map,
                            use_technical_indicator=True,
                            user_defined_feature=False).preprocess_data()

    GBPUSD_train, tech_indicator_list = FeatureEngineer(GBPUSD_df,
                            tech_indicator_params_map = param_map,
                            use_technical_indicator=True,
                            user_defined_feature=False).preprocess_data()

    USDJPY_train, tech_indicator_list = FeatureEngineer(USDJPY_df,
                            tech_indicator_params_map = param_map,
                            use_technical_indicator=True,
                            user_defined_feature=False).preprocess_data()

    USDCHF_train, tech_indicator_list = FeatureEngineer(USDCHF_df,
                            tech_indicator_params_map = param_map,
                            use_technical_indicator=True,
                            user_defined_feature=False).preprocess_data()

    dfs_list = {
        "EURUSD" : EURUSD_train,
        "GBPUSD" : GBPUSD_train,
        "USDJPY" : USDJPY_train,
        "USDCHF" : USDCHF_train
    }

    stock_dimension = len(dfs_list)
    state_space = 1 + 3*stock_dimension + len(tech_indicator_list)*stock_dimension
    print(f"Observation Dimension: {stock_dimension}, State Space: {state_space}")
    model_input_space = 2 + 4*stock_dimension + len(tech_indicator_list)*stock_dimension
    print(f"Input Dimension: {stock_dimension}, State Space: {state_space}")

    return {
        "hmax": 10000, 
        "dfs_list" : dfs_list,
        "initial_amount": 100000, 
        "buy_cost_pct": 0, 
        "sell_cost_pct": 0, 
        "state_space": state_space, 
        "tech_indicator_list": tech_indicator_list, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
    }, len(EURUSD_train), model_input_space

def train_iteration(month: int, year: int):

    if month < 10:
        str_month = "0" + str(month)
    else:
        str_month = str(month)
    monthdata = str_month + '_' + str(year)
    print(monthdata)

    env_kwargs, env_timesteps, model_input_space = create_env_kwargs(monthdata)

    prev_month = month - 1
    if prev_month < 10:
        str_prev_month = "0" + str(prev_month)
    else:
        str_prev_month = str(prev_month)

    bo_iter = 0
    test_month = draw_month(month)
    
    def A2C_train(learning_rate_val, epsilon):

        nonlocal bo_iter
        nonlocal model_input_space
        nonlocal test_month

        start = time.time()

        bo_iter+=1

        e_train_gym = StockTradingEnv(**env_kwargs)
        print(e_train_gym.data)

        env_train, _ = e_train_gym.get_sb_env()
        
        num_episodes = 200
        total_timesteps = num_episodes * env_timesteps

        # if previous month's log data is not found
        if not os.path.isfile(f'results/{str_prev_month}_{year}/BO_logs.json'):
            model_name = "a2c"
            MODELS = {"a2c": A2C}
            MODEL_PARAMS = {"n_steps": 20, "ent_coef": 0.001, "learning_rate": learning_rate_val, 'epsilon': epsilon}
            model_a2c = MODELS[model_name](
                    policy="MlpLstmPolicy",
                    env=env_train,
                    model_input_space=model_input_space,
                    tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
                    verbose=1,
                    policy_kwargs=None,
                    seed = 1,
                    **MODEL_PARAMS,
                )
        # use previous month's data as the first BO probe
        else:
            model_a2c, _ = load_best(f'{str_prev_month}_{year}', model_input_space)
            MODEL_PARAMS = {"n_steps": 20, "ent_coef": 0.001, "learning_rate": learning_rate_val, 'epsilon': epsilon}
            if bo_iter != 1:
                for key,value in MODEL_PARAMS.items():
                    setattr(model_a2c,key,value)

        trained_a2c, hidden_states = model_a2c.learn(total_timesteps=total_timesteps, tb_log_name='a2c')

        # get performance on trained environment 
        env_train, _ = e_train_gym.get_sb_env()
        train_episodes_rewards, _, train_rewards_memory_episodes = evaluate_lstm_rewards(trained_a2c, env_train, model_input_space, monthdata, deterministic=False)
        
        # get performance on sampled environment from 2017
        test_env_kwargs, _, model_input_space =create_env_kwargs(test_month)
        e_test_gym = StockTradingEnv(**test_env_kwargs)
        env_test, _ = e_test_gym.get_sb_env()
        episodes_rewards, _, rewards_memory_episodes = evaluate_lstm_rewards(trained_a2c, env_test, model_input_space, monthdata, deterministic=False)

        combined_episodes_rewards = train_episodes_rewards + episodes_rewards
        combined_rewards_memory = train_rewards_memory_episodes[0] + rewards_memory_episodes[0]

        fig, axs = plt.subplots()

        axs.plot(list(accumulate(rewards_memory_episodes[0])))
        axs.set_title(f"Accumulated rewards (Gains in NOP) against timesteps in {test_month}/17")
        fig.tight_layout()

        fig.savefig(f'plots/bo_results/{monthdata}/iteration_{bo_iter}')

        trained_a2c.save(f"saved_models/{monthdata}/model-{bo_iter}")
        if hidden_states is not None:
            with open(f"saved_models/{monthdata}/hidden_state-{bo_iter}.npy", 'wb') as f:
                np.save(f, hidden_states)

        mean_reward = mean(combined_episodes_rewards)
        print("Mean Episodic Reward : ", mean_reward)

        sortino = sortino_ratio(pd.Series(combined_rewards_memory))
        print("Sortino Ratio :", sortino)

        end = time.time()
        print("Elapsed time: ", end-start)
        
        return sortino

    pbounds = {'learning_rate_val': (0.00001, 0.002), 'epsilon': (1e-06, 2e-05)}

    optimizer = BayesianOptimization(
        f=A2C_train,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    # if previous month data can be found, probe existing parameters
    if os.path.isfile(f'results/{str_prev_month}_{year}/BO_logs.json'):
        _, params = load_best(f'{str_prev_month}_{year}', model_input_space)
        optimizer.probe(
            params=params,
            lazy=True,
        )

    logger = JSONLogger(path=f"results/{monthdata}/BO_logs.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(init_points=1, n_iter=2)


def load_best(monthdata: str, model_input_space: int):
    max_target = None
    index = 0
    target_index = 0
    with open(f'results/{monthdata}/BO_logs.json', "r") as f:
        while True:
            try:
                iteration = next(f)
            except StopIteration:
                        break
            bo_log_data = json.loads(iteration)
            print(bo_log_data)
            if max_target is None:
                max_target = bo_log_data["target"]
                target_params = bo_log_data["params"]
            elif bo_log_data["target"] > max_target:
                max_target = bo_log_data["target"]
                target_params = bo_log_data["params"]
                target_index = index
            index += 1
    print("best_iteration :", target_index)
    print("params:", target_params)
    model = A2C.load(f"saved_models/{monthdata}/model-{target_index+1}", model_input_space)
    return model, target_params

def draw_month(current_month: int):
    months = list(range(1,13))
    months.remove(current_month)
    select = sample(months, 1)
    return select

def train_one():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=config.DOW_30_TICKER,
    ).fetch_data()
    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=True,
                    user_defined_feature = False)

    processed = fe.preprocess_data(df)

    # Training & Trade data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)

    # data normalization
    # feaures_list = list(train.columns)
    # feaures_list.remove('date')
    # feaures_list.remove('tic')
    # feaures_list.remove('close')
    # print(feaures_list)
    # data_normaliser = preprocessing.StandardScaler()
    # train[feaures_list] = data_normaliser.fit_transform(train[feaures_list])
    # trade[feaures_list] = data_normaliser.fit_transform(trade[feaures_list])

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
        1
        + 2 * stock_dimension
        + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001, 
        "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
        
    }

    e_train_gym = StockTradingEnv(df = train, **env_kwargs)

    e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 250, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    model_sac = agent.get_model("sac")
    trained_sac = agent.train_model(model=model_sac, 
        tb_log_name='sac', total_timesteps=80000
    )

    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, test_data=trade, test_env=env_trade, test_obs=obs_trade
    )
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print("==============Get Backtest Results===========")
    perf_stats_all = BackTestStats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")
