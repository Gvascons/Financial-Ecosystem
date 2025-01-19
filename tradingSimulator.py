# coding=utf-8

"""
Goal: Implement a trading simulator to simulate and compare trading strategies.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import sys
import importlib
import pickle
import itertools
import datetime
import json

import numpy as np
import pandas as pd

import random
import torch
import optuna 

from tabulate import tabulate
from tqdm import tqdm
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from tradingEnv import TradingEnv
from tradingPerformance import PerformanceEstimator
from timeSeriesAnalyser import TimeSeriesAnalyser
from pathlib import Path
from TDQN import TDQN



###############################################################################
################################ Global variables #############################
###############################################################################

# Variables defining the default trading horizon
startingDate = '2012-1-1'
endingDate = '2020-1-1'
splitingDate = '2018-1-1'

# Variables defining the default observation and state spaces
stateLength = 30
observationSpace = 1 + (stateLength-1)*4
actionSpace = 2

# Variables setting up the default transaction costs
percentageCosts = [0, 0.1, 0.2]
transactionCosts = percentageCosts[1]/100

# Variables specifying the default capital at the disposal of the trader
money = 100000

# Variables specifying the default general training parameters
bounds = [1, 30]
step = 1
numberOfEpisodes = 100

# Dictionary listing the fictive stocks supported
fictives = {
    'Linear Upward' : 'LINEARUP',
    'Linear Downward' : 'LINEARDOWN',
    'Sinusoidal' : 'SINUSOIDAL',
    'Triangle' : 'TRIANGLE',
}

 # Dictionary listing the 30 stocks considered as testbench
stocks = {
    'Dow Jones' : 'DIA',
    'S&P 500' : 'SPY',
    'NASDAQ 100' : 'QQQ',
    'FTSE 100' : 'EZU',
    'Nikkei 225' : 'EWJ',
    'Google' : 'GOOGL',
    'Apple' : 'AAPL',
    'Facebook' : 'FB',
    'Amazon' : 'AMZN',
    'Microsoft' : 'MSFT',
    'Twitter' : 'TWTR',
    'Nokia' : 'NOK',
    'Philips' : 'PHIA.AS',
    'Siemens' : 'SIE.DE',
    'Baidu' : 'BIDU',
    'Alibaba' : 'BABA',
    'Tencent' : '0700.HK',
    'Sony' : '6758.T',
    'JPMorgan Chase' : 'JPM',
    'HSBC' : 'HSBC',
    'CCB' : '0939.HK',
    'ExxonMobil' : 'XOM',
    'Shell' : 'RDSA.AS',
    'PetroChina' : 'PTR',
    'Tesla' : 'TSLA',
    'Volkswagen' : 'VOW3.DE',
    'Toyota' : '7203.T',
    'Coca Cola' : 'KO',
    'AB InBev' : 'ABI.BR',
    'Kirin' : '2503.T'
}

# Dictionary listing the 5 trading indices considered as testbench
indices = {
    'Dow Jones' : 'DIA',
    'S&P 500' : 'SPY',
    'NASDAQ 100' : 'QQQ',
    'FTSE 100' : 'EZU',
    'Nikkei 225' : 'EWJ'
}

# Dictionary listing the 25 company stocks considered as testbench
companies = {
    'Google' : 'GOOGL',
    'Apple' : 'AAPL',
    'Facebook' : 'FB',
    'Amazon' : 'AMZN',
    'Microsoft' : 'MSFT',
    'Twitter' : 'TWTR',
    'Nokia' : 'NOK',
    'Philips' : 'PHIA.AS',
    'Siemens' : 'SIE.DE',
    'Baidu' : 'BIDU',
    'Alibaba' : 'BABA',
    'Tencent' : '0700.HK',
    'Sony' : '6758.T',
    'JPMorgan Chase' : 'JPM',
    'HSBC' : 'HSBC',
    'CCB' : '0939.HK',
    'ExxonMobil' : 'XOM',
    'Shell' : 'RDSA.AS',
    'PetroChina' : 'PTR',
    'Tesla' : 'TSLA',
    'Volkswagen' : 'VOW3.DE',
    'Toyota' : '7203.T',
    'Coca Cola' : 'KO',
    'AB InBev' : 'ABI.BR',
    'Kirin' : '2503.T'
}

# Dictionary listing the classical trading strategies supported
strategies = {
    'Buy and Hold' : 'BuyAndHold',
    'Sell and Hold' : 'SellAndHold',
    'Trend Following Moving Averages' : 'MovingAveragesTF',
    'Mean Reversion Moving Averages' : 'MovingAveragesMR'
}

# Dictionary listing the AI trading strategies supported
strategiesAI = {
    'TDQN' : 'TDQN',
    'PPO' : 'PPO'
}



###############################################################################
########################### Class TradingSimulator ############################
###############################################################################

class TradingSimulator:
    """
    GOAL: Accurately simulating multiple trading strategies on different stocks
          to analyze and compare their performance.
        
    VARIABLES: /
          
    METHODS:   - displayTestbench: Display consecutively all the stocks
                                   included in the testbench.
               - analyseTimeSeries: Perform a detailled analysis of the stock
                                    market price time series.
               - plotEntireTrading: Plot the entire trading activity, with both
                                    the training and testing phases rendered on
                                    the same graph.
               - simulateNewStrategy: Simulate a new trading strategy on a 
                                      a certain stock of the testbench.
               - simulateExistingStrategy: Simulate an already existing
                                           trading strategy on a certain
                                           stock of the testbench.
               - evaluateStrategy: Evaluate a trading strategy on the
                                   entire testbench.
               - evaluateStock: Compare different trading strategies
                                on a certain stock of the testbench.
    """

    def displayTestbench(self, startingDate=startingDate, endingDate=endingDate):
        """
        GOAL: Display consecutively all the stocks included in the
              testbench (trading indices and companies).
        
        INPUTS: - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
        
        OUTPUTS: /
        """

        # Display the stocks included in the testbench (trading indices)
        for _, stock in indices.items():
            env = TradingEnv(stock, startingDate, endingDate, 0)
            env.render() 

        # Display the stocks included in the testbench (companies)
        for _, stock in companies.items():
            env = TradingEnv(stock, startingDate, endingDate, 0)
            env.render()


    def analyseTimeSeries(self, stockName, startingDate=startingDate, endingDate=endingDate, splitingDate=splitingDate):           
        """
        GOAL: Perform a detailled analysis of the stock market
              price time series.
        
        INPUTS: - stockName: Name of the stock (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
        
        OUTPUTS: /
        """

        # Retrieve the trading stock information
        if(stockName in fictives):
            stock = fictives[stockName]
        elif(stockName in indices):
            stock = indices[stockName]
        elif(stockName in companies):
            stock = companies[stockName]    
        # Error message if the stock specified is not valid or not supported
        else:
            print("The stock specified is not valid, only the following stocks are supported:")
            for stock in fictives:
                print("".join(['- ', stock]))
            for stock in indices:
                print("".join(['- ', stock]))
            for stock in companies:
                print("".join(['- ', stock]))
            raise SystemError("Please check the stock specified.")
        
        # TRAINING DATA
        print("\n\n\nAnalysis of the TRAINING phase time series")
        print("------------------------------------------\n")
        trainingEnv = TradingEnv(stock, startingDate, splitingDate, 0)
        timeSeries = trainingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()

        # TESTING DATA
        print("\n\n\nAnalysis of the TESTING phase time series")
        print("------------------------------------------\n")
        testingEnv = TradingEnv(stock, splitingDate, endingDate, 0)
        timeSeries = testingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()

        # ENTIRE TRADING DATA
        print("\n\n\nAnalysis of the entire time series (both training and testing phases)")
        print("---------------------------------------------------------------------\n")
        tradingEnv = TradingEnv(stock, startingDate, endingDate, 0)
        timeSeries = tradingEnv.data['Close']
        analyser = TimeSeriesAnalyser(timeSeries)
        analyser.timeSeriesDecomposition()
        analyser.stationarityAnalysis()
        analyser.cyclicityAnalysis()


    def plotEntireTrading(self, trainingEnv, testingEnv):
        """
        GOAL: Plot the entire trading activity, with both the training
              and testing phases rendered on the same graph.
        
        INPUTS: - trainingEnv: Trading environment for training.
                - testingEnv: Trading environment for testing.
        
        OUTPUTS: /
        """

        # Artificial trick to assert the continuity of the Money curve
        ratio = trainingEnv.data['Money'][-1]/testingEnv.data['Money'][0]
        testingEnv.data['Money'] = ratio * testingEnv.data['Money']

        # Concatenation of the training and testing trading dataframes
        dataframes = [trainingEnv.data, testingEnv.data]
        data = pd.concat(dataframes)

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        trainingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2)
        testingEnv.data['Close'].plot(ax=ax1, color='blue', lw=2, label='_nolegend_') 
        ax1.plot(data.loc[data['Action'] == 1.0].index, 
                 data['Close'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax1.plot(data.loc[data['Action'] == -1.0].index, 
                 data['Close'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')
        
        # Plot the second graph -> Evolution of the trading capital
        trainingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2)
        testingEnv.data['Money'].plot(ax=ax2, color='blue', lw=2, label='_nolegend_') 
        ax2.plot(data.loc[data['Action'] == 1.0].index, 
                 data['Money'][data['Action'] == 1.0],
                 '^', markersize=5, color='green')   
        ax2.plot(data.loc[data['Action'] == -1.0].index, 
                 data['Money'][data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the vertical line seperating the training and testing datasets
        ax1.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)
        ax2.axvline(pd.Timestamp(splitingDate), color='black', linewidth=2.0)
        
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short", "Train/Test separation"])
        ax2.legend(["Capital", "Long", "Short", "Train/Test separation"])
        
        # Get the figures directory from either environment
        figures_dir = getattr(trainingEnv, 'figures_dir', None) or getattr(testingEnv, 'figures_dir', None)
        
        if figures_dir:
            save_path = os.path.join(figures_dir, f'{str(trainingEnv.marketSymbol)}_TrainingTestingRendering.png')
        else:
            # Fallback to default directory
            save_path = ''.join(['Figs/', str(trainingEnv.marketSymbol), '_TrainingTestingRendering', '.png'])
        
        plt.savefig(save_path)
        plt.close(fig)


    def simulateNewStrategy(self, strategyName, stockName,
                        startingDate=startingDate, endingDate=endingDate, splitingDate=splitingDate,
                        observationSpace=observationSpace, actionSpace=actionSpace, 
                        money=money, stateLength=stateLength, transactionCosts=transactionCosts,
                        bounds=bounds, step=step, numberOfEpisodes=numberOfEpisodes,
                        verbose=True, plotTraining=True, rendering=True, showPerformance=True,
                        saveStrategy=False,
                        PPO_PARAMS=None, min_holding_period=10, max_holding_period=30):  # Add PPO_PARAMS
        """
        Simulate a new trading strategy on a certain stock included in the testbench.
        """
        """
        GOAL: Simulate a new trading strategy on a certain stock included in the
              testbench, with both learning and testing phases.
        
        INPUTS: - strategyName: Name of the trading strategy.
                - stockName: Name of the stock (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - numberOfEpisodes: Number of epsiodes of the RL training phase.
                - verbose: Enable the printing of a simulation feedback.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.
        
        OUTPUTS: - tradingStrategy: Trading strategy simulated.
                 - trainingEnv: Trading environment related to the training phase.
                 - testingEnv: Trading environment related to the testing phase.
        """

        # 1. INITIALIZATION PHASE

        # Retrieve the trading strategy information
        if(strategyName in strategies):
            strategy = strategies[strategyName]
            trainingParameters = [bounds, step]
            ai = False
        elif(strategyName in strategiesAI):
            strategy = strategiesAI[strategyName]
            trainingParameters = [numberOfEpisodes]
            ai = True
        # Error message if the strategy specified is not valid or not supported
        else:
            print("The strategy specified is not valid, only the following strategies are supported:")
            for strategy in strategies:
                print("".join(['- ', strategy]))
            for strategy in strategiesAI:
                print("".join(['- ', strategy]))
            raise SystemError("Please check the trading strategy specified.")

        # Retrieve the trading stock information
        if(stockName in fictives):
            stock = fictives[stockName]
        elif(stockName in indices):
            stock = indices[stockName]
        elif(stockName in companies):
            stock = companies[stockName]    
        # Error message if the stock specified is not valid or not supported
        else:
            print("The stock specified is not valid, only the following stocks are supported:")
            for stock in fictives:
                print("".join(['- ', stock]))
            for stock in indices:
                print("".join(['- ', stock]))
            for stock in companies:
                print("".join(['- ', stock]))
            raise SystemError("Please check the stock specified.")


        # 2. TRAINING PHASE

        # Initialize the trading environment associated with the training phase
        trainingEnv = TradingEnv(stock, startingDate, splitingDate, money, stateLength, transactionCosts, min_holding_period=min_holding_period, max_holding_period=max_holding_period)

        # Instanciate the strategy classes
         # Instantiate the strategy classes
        if ai:
            strategyModule = importlib.import_module(str(strategy))
            className = getattr(strategyModule, strategy)
            if strategy == 'PPO':
                tradingStrategy = className(observationSpace, actionSpace, PPO_PARAMS, marketSymbol=stock)
            else:
                tradingStrategy = className(observationSpace, actionSpace, marketSymbol=stock)
        else:
            strategyModule = importlib.import_module('classicalStrategy')
            className = getattr(strategyModule, strategy)
            tradingStrategy = className()

        # Training of the trading strategy
        trainingEnv = tradingStrategy.training(trainingEnv, trainingParameters=trainingParameters,
                                               verbose=verbose, rendering=rendering,
                                               plotTraining=plotTraining, showPerformance=showPerformance)

        
        # 3. TESTING PHASE

        # Initialize the trading environment associated with the testing phase
        testingEnv = TradingEnv(stock, splitingDate, endingDate, money, stateLength, transactionCosts)

        # Testing of the trading strategy
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=rendering, showPerformance=showPerformance)
            
        # Show the entire unified rendering of the training and testing phases
        if rendering:
            self.plotEntireTrading(trainingEnv, testingEnv)


        # 4. TERMINATION PHASE

        # If required, save the trading strategy with Pickle
        if(saveStrategy):
            fileName = "".join(["Strategies/", strategy, "_", stock, "_", startingDate, "_", splitingDate])
            if ai:
                tradingStrategy.saveModel(fileName)
            else:
                fileHandler = open(fileName, 'wb') 
                pickle.dump(tradingStrategy, fileHandler)

        # Return of the trading strategy simulated and of the trading environments backtested
        return tradingStrategy, trainingEnv, testingEnv

    
    def simulateExistingStrategy(self, strategyName, stockName,
                             startingDate=startingDate, endingDate=endingDate, splitingDate=splitingDate,
                             observationSpace=observationSpace, actionSpace=actionSpace, 
                             money=money, stateLength=stateLength, transactionCosts=transactionCosts,
                             rendering=True, showPerformance=True,
                             PPO_PARAMS=None):  # Add PPO_PARAMS
        """
        GOAL: Simulate an already existing trading strategy on a certain
              stock of the testbench, the strategy being loaded from the
              strategy dataset. There is no training phase, only a testing
              phase.
        
        INPUTS: - strategyName: Name of the trading strategy.
                - stockName: Name of the stock (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
        
        OUTPUTS: - tradingStrategy: Trading strategy simulated.
                 - trainingEnv: Trading environment related to the training phase.
                 - testingEnv: Trading environment related to the testing phase.
        """

        # 1. INITIALIZATION PHASE

        # Retrieve the trading strategy information
        if(strategyName in strategies):
            strategy = strategies[strategyName]
            ai = False
        elif(strategyName in strategiesAI):
            strategy = strategiesAI[strategyName]
            ai = True
        # Error message if the strategy specified is not valid or not supported
        else:
            print("The strategy specified is not valid, only the following strategies are supported:")
            for strategy in strategies:
                print("".join(['- ', strategy]))
            for strategy in strategiesAI:
                print("".join(['- ', strategy]))
            raise SystemError("Please check the trading strategy specified.")

        # Retrieve the trading stock information
        if(stockName in fictives):
            stock = fictives[stockName]
        elif(stockName in indices):
            stock = indices[stockName]
        elif(stockName in companies):
            stock = companies[stockName]    
        # Error message if the stock specified is not valid or not supported
        else:
            print("The stock specified is not valid, only the following stocks are supported:")
            for stock in fictives:
                print("".join(['- ', stock]))
            for stock in indices:
                print("".join(['- ', stock]))
            for stock in companies:
                print("".join(['- ', stock]))
            raise SystemError("Please check the stock specified.")
        

        # 2. LOADING PHASE

        # Check that the strategy to load exists in the strategy dataset
        fileName = "".join(["Strategies/", strategy, "_", stock, "_", startingDate, "_", splitingDate])
        exists = os.path.isfile(fileName)
        # If affirmative, load the trading strategy
        if exists:
            if ai:
                strategyModule = importlib.import_module(strategy)
                className = getattr(strategyModule, strategy)
                if strategy == 'PPO':
                    tradingStrategy = className(observationSpace, actionSpace, PPO_PARAMS, marketSymbol=stock)
                else:
                    tradingStrategy = className(observationSpace, actionSpace)
                tradingStrategy.loadModel(fileName)
            else:
                fileHandler = open(fileName, 'rb') 
                tradingStrategy = pickle.load(fileHandler)
        else:
            raise SystemError("The trading strategy specified does not exist, please provide a valid one.")


        # 3. TESTING PHASE

        # Initialize the trading environments associated with the testing phase
        trainingEnv = TradingEnv(stock, startingDate, splitingDate, money, stateLength, transactionCosts)
        testingEnv = TradingEnv(stock, splitingDate, endingDate, money, stateLength, transactionCosts)

        # Testing of the trading strategy
        trainingEnv = tradingStrategy.testing(trainingEnv, trainingEnv, rendering=rendering, showPerformance=showPerformance)
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=rendering, showPerformance=showPerformance)

        # Show the entire unified rendering of the training and testing phases
        if rendering:
            self.plotEntireTrading(trainingEnv, testingEnv)

        return tradingStrategy, trainingEnv, testingEnv
    
    def optimizeHyperparameters(self, strategyName, stockName,
                            startingDate=startingDate, endingDate=endingDate, splitingDate=splitingDate,
                            observationSpace=observationSpace, actionSpace=actionSpace, 
                            money=money, stateLength=stateLength, transactionCosts=transactionCosts,
                            numberOfEpisodes=100, n_trials=50, rendering=False):
        """
        Optimize hyperparameters for the specified strategy and stock.
        """
        if strategyName != 'PPO':
            raise NotImplementedError("Hyperparameter optimization is currently implemented only for PPO.")
        
        # Retrieve the trading stock information
        if(stockName in fictives):
            stock = fictives[stockName]
        elif(stockName in indices):
            stock = indices[stockName]
        elif(stockName in companies):
            stock = companies[stockName]
        else:
            print("The stock specified is not valid, only the following stocks are supported:")
            for s in fictives:
                print("".join(['- ', s]))
            for s in indices:
                print("".join(['- ', s]))
            for s in companies:
                print("".join(['- ', s]))
            raise SystemError("Please check the stock specified.")
        
        # Define the objective function for Optuna
        def objective(trial):
            try:
                # Sugerir períodos mínimo e máximo de holding
                min_holding_period = trial.suggest_int('min_holding_period', 1, 30) ############################
                max_holding_period = trial.suggest_int('max_holding_period', 50, 200) ############################
                # Suggest number of LSTM layers
                lstm_layers = trial.suggest_int('LSTM_LAYERS', 1, 3)

                # Suggest dropout only if num_layers > 1
                if lstm_layers > 1:
                    lstm_dropout = trial.suggest_float('LSTM_DROPOUT', 0.0, 0.5)
                else:
                    lstm_dropout = 0.0  # Set dropout to zero when num_layers is 1

                # Suggest other hyperparameters
                PPO_PARAMS = {
                    'CLIP_EPSILON': trial.suggest_float('CLIP_EPSILON', 0.1, 0.3),
                    'VALUE_LOSS_COEF': trial.suggest_float('VALUE_LOSS_COEF', 0.1, 1.0),
                    'ENTROPY_COEF': trial.suggest_float('ENTROPY_COEF', 0.0, 0.05),
                    'PPO_EPOCHS': trial.suggest_int('PPO_EPOCHS', 1, 10),
                    'BATCH_SIZE': trial.suggest_int('BATCH_SIZE', 32, 256, log=True),
                    'GAMMA': trial.suggest_float('GAMMA', 0.9, 0.9999),
                    'GAE_LAMBDA': trial.suggest_float('GAE_LAMBDA', 0.8, 1.0),
                    'LEARNING_RATE': trial.suggest_float('LEARNING_RATE', 1e-5, 1e-3, log=True),
                    'MAX_GRAD_NORM': trial.suggest_float('MAX_GRAD_NORM', 0.1, 1.0),
                    'HIDDEN_SIZE': trial.suggest_categorical('HIDDEN_SIZE', [64, 128, 256, 512]),
                    'MEMORY_SIZE': 10000,
                    'LSTM_HIDDEN_SIZE': trial.suggest_categorical('LSTM_HIDDEN_SIZE', [64, 128, 256]),
                    'LSTM_LAYERS': lstm_layers,
                    'LSTM_DROPOUT': lstm_dropout,
                }

                # Generate a unique run_id using trial number
                run_id = f"TRIAL_{trial.number}_PPO_{stock}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

                # Initialize the trading strategy with suggested hyperparameters and run_id
                strategyModule = importlib.import_module('PPO')
                className = getattr(strategyModule, 'PPO')
                tradingStrategy = className(observationSpace, actionSpace, PPO_PARAMS, marketSymbol=stock, run_id=run_id)

                # Set seeds for reproducibility
                seed = trial.number
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)

                # Initialize the trading environment
                trainingEnv = TradingEnv(stock, startingDate, splitingDate, money, stateLength, transactionCosts, min_holding_period=min_holding_period, max_holding_period=max_holding_period)

                # Train the strategy
                trainingParameters = [numberOfEpisodes]
                trainingEnv = tradingStrategy.training(
                    trainingEnv, trainingParameters=trainingParameters,
                    verbose=False, rendering=False,
                    plotTraining=False, showPerformance=False
                )

                # Testing phase
                testingEnv = TradingEnv(stock, splitingDate, endingDate, money, stateLength, transactionCosts, min_holding_period=min_holding_period, max_holding_period=max_holding_period)
                testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=False, showPerformance=False)

                # Evaluate performance
                analyser = PerformanceEstimator(testingEnv.data)
                performance = analyser.computeSharpeRatio()

                # Optuna minimizes the objective, so return negative Sharpe Ratio
                return -performance
        
            except Exception as e:
                print(f"Trial {trial.number} failed with exception: {e}")
                return float('inf')  # Return a high value to indicate failure
            
        # Create the Optuna study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        # Get the best hyperparameters
        best_params = study.best_params
        print("Best hyperparameters:", best_params)

        # Create a subfolder for this stock if you want to store everything
        stock_subfolder = f"models/{stock}"
        Path(stock_subfolder).mkdir(parents=True, exist_ok=True)

        # Save best_params to a JSON file for future usage
        with open(os.path.join(stock_subfolder, "best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)

        min_holding_period = best_params['min_holding_period']
        max_holding_period = best_params['max_holding_period']

        # Use the best hyperparameters to train the final model
        PPO_PARAMS = {
            'CLIP_EPSILON': best_params['CLIP_EPSILON'],
            'VALUE_LOSS_COEF': best_params['VALUE_LOSS_COEF'],
            'ENTROPY_COEF': best_params['ENTROPY_COEF'],
            'PPO_EPOCHS': best_params['PPO_EPOCHS'],
            'BATCH_SIZE': best_params['BATCH_SIZE'],
            'GAMMA': best_params['GAMMA'],
            'GAE_LAMBDA': best_params['GAE_LAMBDA'],
            'LEARNING_RATE': best_params['LEARNING_RATE'],
            'MAX_GRAD_NORM': best_params['MAX_GRAD_NORM'],
            'HIDDEN_SIZE': best_params['HIDDEN_SIZE'],
            'MEMORY_SIZE': 10000,
            'LSTM_HIDDEN_SIZE': best_params['LSTM_HIDDEN_SIZE'],
            'LSTM_LAYERS': best_params['LSTM_LAYERS'],
            'LSTM_DROPOUT': best_params.get('LSTM_DROPOUT', 0.0),  # Use get() with default 0.0
        }

        # Increase the number of episodes for final training
        final_number_of_episodes = 100  # Adjust as needed

        # Generate a unique run_id for the final model
        run_id = f"run_PPO_{stock}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Train the final model with the best hyperparameters
        trainingEnv = TradingEnv(stock, startingDate, splitingDate, money, stateLength, transactionCosts, min_holding_period=min_holding_period, max_holding_period=max_holding_period)
        strategyModule = importlib.import_module('PPO')
        className = getattr(strategyModule, 'PPO')

        # Pass the run_id when initializing the PPO agent
        tradingStrategy = className(observationSpace, actionSpace, PPO_PARAMS, marketSymbol=stock, run_id=run_id)

        trainingParameters = [final_number_of_episodes]
        trainingEnv = tradingStrategy.training(trainingEnv, trainingParameters=trainingParameters,
                                           verbose=True, rendering=rendering,
                                           plotTraining=True, showPerformance=True)

        # Test the final model
        testingEnv = TradingEnv(stock, splitingDate, endingDate, money, stateLength, transactionCosts, min_holding_period=min_holding_period, max_holding_period=max_holding_period)
        testingEnv = tradingStrategy.testing(trainingEnv, testingEnv, rendering=rendering, showPerformance=True)

        # Show the entire unified rendering of the training and testing phases
        if rendering:
            self.plotEntireTrading(trainingEnv, testingEnv)

        best_model_path = os.path.join(stock_subfolder, "my_best_ppo_model.pt")

        # after the final training completes:
        torch.save(tradingStrategy.network.state_dict(), best_model_path)

        return tradingStrategy, trainingEnv, testingEnv
    

    def evaluateStrategy(self, strategyName,
                         startingDate=startingDate, endingDate=endingDate, splitingDate=splitingDate,
                         observationSpace=observationSpace, actionSpace=actionSpace, 
                         money=money, stateLength=stateLength, transactionCosts=transactionCosts,
                         bounds=bounds, step=step, numberOfEpisodes=numberOfEpisodes,
                         verbose=False, plotTraining=False, rendering=False, showPerformance=False,
                         saveStrategy=False):
        """
        GOAL: Evaluate the performance of a trading strategy on the entire
              testbench of stocks designed.
        
        INPUTS: - strategyName: Name of the trading strategy.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - observationSpace: Size of the RL observation space.
                - actionSpace: Size of the RL action space.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - numberOfEpisodes: Number of epsiodes of the RL training phase.
                - verbose: Enable the printing of simulation feedbacks.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.
        
        OUTPUTS: - performanceTable: Table summarizing the performance of
                                     a trading strategy.
        """

        # Initialization of some variables
        performanceTable = [["Profit & Loss (P&L)"], ["Annualized Return"], ["Annualized Volatility"], ["Sharpe Ratio"], ["Sortino Ratio"], ["Maximum DrawDown"], ["Maximum DrawDown Duration"], ["Profitability"], ["Ratio Average Profit/Loss"], ["Skewness"]]
        headers = ["Performance Indicator"]

        # Loop through each stock included in the testbench (progress bar)
        print("Trading strategy evaluation progression:")
        #for stock in tqdm(itertools.chain(indices, companies)):
        for stock in tqdm(stocks):

            # Simulation of the trading strategy on the current stock
            try:
                # Simulate an already existing trading strategy on the current stock
                _, _, testingEnv = self.simulateExistingStrategy(strategyName, stock, startingDate, endingDate, splitingDate, observationSpace, actionSpace, money, stateLength, transactionCosts, rendering, showPerformance)
            except SystemError:
                # Simulate a new trading strategy on the current stock
                _, _, testingEnv = self.simulateNewStrategy(strategyName, stock, startingDate, endingDate, splitingDate, observationSpace, actionSpace, money, stateLength, transactionCosts, bounds, step, numberOfEpisodes, verbose, plotTraining, rendering, showPerformance, saveStrategy)

            # Retrieve the trading performance associated with the trading strategy
            analyser = PerformanceEstimator(testingEnv.data)
            performance = analyser.computePerformance()
            
            # Get the required format for the display of the performance table
            headers.append(stock)
            for i in range(len(performanceTable)):
                performanceTable[i].append(performance[i][1])

        # Display the performance table computed
        tabulation = tabulate(performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)

        # Computation of the average Sharpe Ratio (default performance indicator)
        sharpeRatio = np.mean([float(item) for item in performanceTable[3][1:]])
        print("Average Sharpe Ratio: " + "{0:.3f}".format(sharpeRatio))

        return performanceTable


    def evaluateStock(self, stockName,
                      startingDate=startingDate, endingDate=endingDate, splitingDate=splitingDate,
                      observationSpace=observationSpace, actionSpace=actionSpace,  
                      money=money, stateLength=stateLength, transactionCosts=transactionCosts,
                      bounds=bounds, step=step, numberOfEpisodes=numberOfEpisodes,
                      verbose=False, plotTraining=False, rendering=False, showPerformance=False,
                      saveStrategy=False):

        """
        GOAL: Simulate and compare the performance achieved by all the supported
              trading strategies on a certain stock of the testbench.
        
        INPUTS: - stockName: Name of the stock (in the testbench).
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - splitingDate: Spliting date between the training dataset
                                and the testing dataset.
                - money: Initial capital at the disposal of the agent.
                - stateLength: Length of the trading agent state.
                - transactionCosts: Additional costs incurred while trading
                                    (e.g. 0.01 <=> 1% of transaction costs).
                - bounds: Bounds of the parameter search space (training).
                - step: Step of the parameter search space (training).
                - numberOfEpisodes: Number of epsiodes of the RL training phase.
                - verbose: Enable the printing of a simulation feedback.
                - plotTraining: Enable the plotting of the training results.
                - rendering: Enable the rendering of the trading environment.
                - showPerformance: Enable the printing of a table summarizing
                                   the trading strategy performance.
                - saveStrategy: Enable the saving of the trading strategy.
        
        OUTPUTS: - performanceTable: Table summarizing the performance of
                                     a trading strategy.
        """

        # Initialization of some variables
        performanceTable = [["Profit & Loss (P&L)"], ["Annualized Return"], ["Annualized Volatility"], ["Sharpe Ratio"], ["Sortino Ratio"], ["Maximum DrawDown"], ["Maximum DrawDown Duration"], ["Profitability"], ["Ratio Average Profit/Loss"], ["Skewness"]]
        headers = ["Performance Indicator"]

        # Loop through all the trading strategies supported (progress bar)
        print("Trading strategies evaluation progression:")
        for strategy in tqdm(itertools.chain(strategies, strategiesAI)):

            # Simulation of the current trading strategy on the stock
            try:
                # Simulate an already existing trading strategy on the stock
                _, _, testingEnv = self.simulateExistingStrategy(strategy, stockName, startingDate, endingDate, splitingDate, observationSpace, actionSpace, money, stateLength, transactionCosts, rendering, showPerformance)
            except SystemError:
                # Simulate a new trading strategy on the stock
                _, _, testingEnv = self.simulateNewStrategy(strategy, stockName, startingDate, endingDate, splitingDate, observationSpace, actionSpace, money, stateLength, transactionCosts, bounds, step, numberOfEpisodes, verbose, plotTraining, rendering, showPerformance, saveStrategy)

            # Retrieve the trading performance associated with the trading strategy
            analyser = PerformanceEstimator(testingEnv.data)
            performance = analyser.computePerformance()
            
            # Get the required format for the display of the performance table
            headers.append(strategy)
            for i in range(len(performanceTable)):
                performanceTable[i].append(performance[i][1])

        # Display the performance table
        tabulation = tabulate(performanceTable, headers, tablefmt="fancy_grid", stralign="center")
        print(tabulation)

        return performanceTable

    def runSavedModel(self,
                      model_path,
                      PPO_PARAMS,
                      stockSymbol,
                      startingDate,
                      splitingDate,
                      endingDate,
                      observationSpace,
                      actionSpace,
                      money,
                      stateLength,
                      transactionCosts,
                      deterministic=True,
                      rendering=False,
                      showPerformance=True):
        """
        All we are doing is creating two TradingEnv objects:
        1) A "training environment" (2012 to 2018) for the sole purpose of computing normalization coefficients (e.g., mean/std) so your final model sees states scaled exactly the same way as it did before.
        2) A "test environment" (2018 to 2020) where you run the inference loop deterministically to replicate your final testing run.
        There is no call to any training loop or backprop step in runSavedModel. 

        Load a previously saved PPO policy and run a deterministic test 
        just like the final step in your 'simulateNewStrategy' method.
        We do:
          1) Training env from (startingDate -> splitingDate) 
             to compute normalization coefficients
          2) Testing env from (splitingDate -> endingDate)
             to run the final deterministic inference

        Args:
            model_path (str): Path to the saved model (e.g. "my_best_ppo_model.pt").
            PPO_PARAMS (dict): Same PPO hyperparameters used in training 
                               (HIDDEN_SIZE, LSTM_HIDDEN_SIZE, etc.).
            stockSymbol (str): Ticker or synthetic name (e.g. "AAPL", "LINEARUP", etc.).
            startingDate (str): Start date of the trading horizon (e.g. "2019-01-01").
            endingDate (str): End date of the trading horizon (e.g. "2020-01-01").
            splitingDate (str): Not strictly needed for the test env, but 
                                if you want to do trainingEnv vs. testingEnv normalization,
                                you can. Otherwise, you may skip or set it to None.
            observationSpace (int): Dimension of the observation space.
            actionSpace (int): Dimension of the action space.
            money (float): Initial capital.
            stateLength (int): How many timesteps the environment includes in each state.
            transactionCosts (float): Transaction fee fraction (e.g. 0.001 => 0.1%).
            deterministic (bool): Whether to use argmax action selection for testing.
            rendering (bool): Whether to render the final trades chart to a .png file.
            showPerformance (bool): If True, runs a PerformanceEstimator at the end 
                                    to display metrics (PnL, Sharpe, etc.).

        Returns:
            TradingEnv: The environment instance after the test run (contains data for analysis).
        """
        import torch
        from PPO import PPO
        from tradingEnv import TradingEnv

        # Ensure MEMORY_SIZE exists in PPO_PARAMS
        PPO_PARAMS.setdefault('MEMORY_SIZE', 10000)

        # 1) Create a training environment for normalization
        trainingEnv = TradingEnv(
            marketSymbol=stockSymbol, 
            startingDate=startingDate,
            endingDate=splitingDate,
            money=money,
            stateLength=stateLength,
            transactionCosts=transactionCosts,
            min_holding_period=PPO_PARAMS.get('min_holding_period', 1),
            max_holding_period=PPO_PARAMS.get('max_holding_period', 200)
        )

        # 2) Create the testing environment for the final run
        testingEnv = TradingEnv(
            marketSymbol=stockSymbol,
            startingDate=splitingDate, 
            endingDate=endingDate,
            money=money,
            stateLength=stateLength,
            transactionCosts=transactionCosts,
            min_holding_period=PPO_PARAMS.get('min_holding_period', 1),
            max_holding_period=PPO_PARAMS.get('max_holding_period', 200)
        )

        # 3) Build a new PPO agent with the same hyperparameters
        agent = PPO(observationSpace, actionSpace, PPO_PARAMS, marketSymbol=stockSymbol, run_id=None)

        # 4) Load the saved model weights and set to eval mode
        agent.network.eval()

        # Safely load only the state dict
        state_dict = torch.load(model_path, weights_only=True)
        agent.network.load_state_dict(state_dict)

        # 5) If you use normalization, compute it from the trainingEnv
        coefficients = agent.getNormalizationCoefficients(trainingEnv)

        # 6) Run deterministic inference on the testing environment
        state = testingEnv.reset()
        done = False

        while not done:
            # Apply the same normalization as you do in training
            state = agent.processState(state, coefficients)

            # Deterministic => picks argmax
            action, _, _ = agent.select_action(state, deterministic=deterministic)
            next_state, reward, done, _ = testingEnv.step(action)
            state = next_state

        # If requested, render the final test chart in the stock's subfolder
        if rendering:
            # Create subfolder for this stock's figures if it doesn't exist
            fig_subfolder = f"Figs/{stockSymbol}_Loaded"
            Path(fig_subfolder).mkdir(parents=True, exist_ok=True)
            
            # Save the figure in the stock's subfolder
            fig_path = os.path.join(fig_subfolder, f"{stockSymbol}_Inference_Rendering.png")
            testingEnv.render(save_path=fig_path)  # You might need to modify TradingEnv.render() to accept save_path

        # If requested, show performance metrics for the final test
        if showPerformance:
            from tradingPerformance import PerformanceEstimator
            analyser = PerformanceEstimator(testingEnv.data)
            analyser.displayPerformance(name='Loaded_PPO', phase='testing')

        return testingEnv