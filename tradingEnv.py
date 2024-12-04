# coding=utf-8

"""
Goal: Implement a trading environment compatible with OpenAI Gym.
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import gym
import math
import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None

from matplotlib import pyplot as plt

from dataDownloader import AlphaVantage
from dataDownloader import YahooFinance
from dataDownloader import CSVHandler
from fictiveStockGenerator import StockGenerator

###############################################################################
################################ Global variables #############################
###############################################################################

# Boolean handling the saving of the stock market data downloaded
saving = True

# Variable related to the fictive stocks supported
fictiveStocks = ('LINEARUP', 'LINEARDOWN', 'SINUSOIDAL', 'TRIANGLE')

###############################################################################
############################## Class TradingEnv ###############################
###############################################################################

class TradingEnv(gym.Env):
    """
    GOAL: Implement a custom trading environment compatible with OpenAI Gym.

    VARIABLES:
        - data: Dataframe monitoring the trading activity.
        - state: RL state to be returned to the RL agent.
        - reward: RL reward to be returned to the RL agent.
        - done: RL episode termination signal.
        - t: Current trading time step.
        - marketSymbol: Stock market symbol.
        - startingDate: Beginning of the trading horizon.
        - endingDate: Ending of the trading horizon.
        - stateLength: Number of trading time steps included in the state.
        - numberOfShares: Number of shares currently owned by the agent.
        - transactionCosts: Transaction costs associated with the trading
                            activity (e.g. 0.01 is 1% of loss).
        - holding_period: Number of days the agent has held the current position.

    METHODS:
        - __init__: Object constructor initializing the trading environment.
        - reset: Perform a soft reset of the trading environment.
        - step: Transition to the next trading time step.
        - render: Illustrate graphically the trading environment.
    """

    def __init__(self, marketSymbol, startingDate, endingDate, money, stateLength=30,
             transactionCosts=0, startingPoint=0, min_holding_period=10, max_holding_period=100):
        """
        GOAL: Object constructor initializing the trading environment by setting up
              the trading activity dataframe as well as other important variables.

        INPUTS:
            - marketSymbol: Stock market symbol.
            - startingDate: Beginning of the trading horizon.
            - endingDate: Ending of the trading horizon.
            - money: Initial amount of money at the disposal of the agent.
            - stateLength: Number of trading time steps included in the RL state.
            - transactionCosts: Transaction costs associated with the trading
                                activity (e.g. 0.01 is 1% of loss).
            - startingPoint: Optional starting point (iteration) of the trading activity.

        OUTPUTS: /
        """

        # Move stateLength initialization to the top
        self.stateLength = stateLength
        # Initialize holding period constraints
        self.min_holding_period = min_holding_period
        self.max_holding_period = max_holding_period
        self.holding_period = 0  # Days the current position has been held

        # CASE 1: Fictive stock generation
        if(marketSymbol in fictiveStocks):
            stockGeneration = StockGenerator()
            if(marketSymbol == 'LINEARUP'):
                self.data = stockGeneration.linearUp(startingDate, endingDate)
            elif(marketSymbol == 'LINEARDOWN'):
                self.data = stockGeneration.linearDown(startingDate, endingDate)
            elif(marketSymbol == 'SINUSOIDAL'):
                self.data = stockGeneration.sinusoidal(startingDate, endingDate)
            else:
                self.data = stockGeneration.triangle(startingDate, endingDate)

        # CASE 2: Real stock loading
        else:
            # Check if the stock market data is already present in the database
            csvConverter = CSVHandler()
            csvName = "".join(['Data/', marketSymbol, '_', startingDate, '_', endingDate])
            exists = os.path.isfile(csvName + '.csv')

            # If affirmative, load the stock market data from the database
            if(exists):
                self.data = csvConverter.CSVToDataframe(csvName)
            # Otherwise, download the stock market data from Yahoo Finance and save it in the database
            else:
                downloader1 = YahooFinance()
                downloader2 = AlphaVantage()
                try:
                    self.data = downloader1.getDailyData(marketSymbol, startingDate, endingDate)
                except:
                    self.data = downloader2.getDailyData(marketSymbol, startingDate, endingDate)

                if saving == True:
                    csvConverter.dataframeToCSV(csvName, self.data)

        # Interpolate in case of missing data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)

        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Prepare the state data (including technical indicators)
        self.features = ['Close', 'Low', 'High', 'Volume', 'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
                         'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist', 'BB_Middle', 'BB_Upper',
                         'BB_Lower', 'ATR_14', 'OBV']

        # Ensure that all features are present in the data
        for feature in self.features:
            if feature not in self.data.columns:
                self.data[feature] = 0.0  # or handle accordingly

        # Set the RL variables common to every OpenAI gym environments
        self.state = self.getState(0)
        self.reward = 0.
        self.done = 0

        # Set additional variables related to the trading activity
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1

        # Initialize holding period
        self.holding_period = 0  # Days the current position has been held

        # If required, set a custom starting point for the trading activity
        if startingPoint:
            self.setStartingPoint(startingPoint)

        # Add this near the beginning of __init__
        self.action_space = 2  # 0 for short, 1 for long
        self.observation_space = len(self.features) + 1  # Number of features plus position

    def getState(self, t):
        """
        GOAL: Construct the state representation for the agent.

        INPUTS: - t: Current time step.

        OUTPUTS: - state: List containing the state information.
        """
        # Initialize state list
        state = []

        # Get data for each feature
        for feature in self.features:
            # Get the feature data for the window
            feature_data = self.data[feature][max(0, t - self.stateLength): t].values
            # Ensure the data is the correct length
            if len(feature_data) < self.stateLength:
                # Pad with zeros if necessary
                padding = np.zeros(self.stateLength - len(feature_data))
                feature_data = np.concatenate([padding, feature_data])
            state.append(feature_data)

        # Add the current position, repeated to match the sequence length
        position = np.full(self.stateLength, self.data['Position'][max(0, t - 1)])
        state.append(position)

        # Convert to numpy array with proper shape
        state = np.array(state, dtype=np.float32)  # Shape: [n_features, sequence_length]

        return state

    def reset(self):
        """
        GOAL: Perform a soft reset of the trading environment.

        INPUTS: /

        OUTPUTS: - state: RL state returned to the trading strategy.
        """

        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Reset the RL variables common to every OpenAI gym environments
        self.state = self.getState(self.stateLength)
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0
        self.holding_period = 0  # Reset holding period

        return self.state

    def computeLowerBound(self, cash, numberOfShares, price):
        """
        GOAL: Compute the lower bound of the complete RL action space,
              i.e. the minimum number of share to trade.

        INPUTS: - cash: Value of the cash owned by the agent.
                - numberOfShares: Number of shares owned by the agent.
                - price: Last price observed.

        OUTPUTS: - lowerBound: Lower bound of the RL action space.
        """

        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound

    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (either long or short).

        INPUTS: - action: Trading decision (1 = long, 0 = short).

        OUTPUTS: - state: RL state to be returned to the RL agent.
                 - reward: RL reward to be returned to the RL agent.
                 - done: RL episode termination signal (boolean).
                 - info: Additional information returned to the RL agent.
        """

        # Setting of some local variables
        t = self.t
        numberOfShares = self.numberOfShares
        self.customReward = False  # Reset custom reward flag

        current_position = self.data['Position'][t - 1]  # Previous position

        # Check if the agent is trying to change position
        trying_to_change = (action != (1 if current_position == 1 else 0))

        # Enforce minimum holding period
        if self.holding_period < self.min_holding_period:
            if trying_to_change:
                # Prevent position change
                action = 1 if current_position == 1 else 0
        # Enforce maximum holding period
        elif self.holding_period >= self.max_holding_period:
            if not trying_to_change:
                # Force position change
                action = 0 if current_position == 1 else 1
                # Reset holding period after forced change
                self.holding_period = 0

        # Update holding period
        if action == (1 if current_position == 1 else 0):
            self.holding_period += 1
        else:
            self.holding_period = 1  # Reset holding period after position change

        # Proceed with the original step logic

        # CASE 1: LONG POSITION
        if(action == 1):
            self.data['Position'][t] = 1
            # Case a: Long -> Long
            if(current_position == 1):
                self.data['Cash'][t] = self.data['Cash'][t - 1]
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
            # Case b: No position -> Long
            elif(current_position == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1
            # Case c: Short -> Long
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = 1

        # CASE 2: SHORT POSITION
        elif(action == 0):
            self.data['Position'][t] = -1
            # Case a: Short -> Short
            if(current_position == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                    self.customReward = True
            # Case b: No position -> Short
            elif(current_position == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1
            # Case c: Long -> Short
            else:
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1

        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short).")

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][self.t] = self.data['Holdings'][self.t] + self.data['Cash'][self.t]
        self.data['Returns'][self.t] = (self.data['Money'][self.t] - self.data['Money'][self.t-1])/self.data['Money'][self.t-1]

        # Calculate reward using the enhanced method
        self.reward = self.calculate_reward()

        # Transition to the next trading time step
        self.t = self.t + 1
        if(self.t == self.data.shape[0]):
            self.done = 1
            self.state = self.getState(self.t - 1)  # Last state
        else:
            self.state = self.getState(self.t)

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, {}

    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.

        INPUTS: /

        OUTPUTS: /
        """
        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index,
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='green')
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index,
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='red')

        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long",  "Short"])
        ax2.legend(["Capital", "Long", "Short"])

        plt.savefig(''.join(['Figs/', str(self.marketSymbol), '_Rendering', '.png']))
        plt.close(fig)

    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.

        INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.

        OUTPUTS: /
        """

        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))

        # Set the RL variables common to every OpenAI gym environments
        self.state = self.getState(self.t)
        if(self.t == self.data.shape[0]):
            self.done = 1

    def calculate_reward(self):
        """
        Enhanced reward function that considers multiple trading objectives while
        maintaining compatibility with the existing trading logic.
        """
        # If it's a custom reward case (specific short position scenario)
        if self.customReward:
            return (self.data['Close'][self.t-1] - self.data['Close'][self.t])/self.data['Close'][self.t-1]

        # Standard case: Enhanced reward calculation
        # Base return (already includes transaction costs)
        base_return = self.data['Returns'][self.t]

        # Calculate risk-adjusted components
        lookback = min(self.stateLength, self.t)
        returns = self.data['Returns'][max(0, self.t - lookback):self.t]

        # Downside risk (Sortino ratio component)
        downside_returns = returns[returns < 0]
        downside_risk = np.std(downside_returns) if len(downside_returns) > 0 else 0
        risk_penalty = -0.1 * downside_risk

        # Position holding incentive (reduce excessive trading)
        position_change = abs(self.data['Position'][self.t] - self.data['Position'][self.t-1])
        trading_penalty = -0.001 * position_change

        # Trend alignment bonus (reward for following the trend)
        price_trend = (self.data['Close'][self.t] - self.data['Close'][self.t-1]) / self.data['Close'][self.t-1]
        position = self.data['Position'][self.t]
        trend_alignment = 0.1 * price_trend * position  # Positive when position aligns with trend

        # Combine components with weights
        reward = (
            1.0 * base_return +  # Main return component
            0.3 * risk_penalty +  # Risk adjustment
            0.2 * trading_penalty +  # Trading frequency penalty
            0.2 * trend_alignment  # Trend alignment bonus
        )

        # Scale reward for better learning stability
        reward = np.clip(reward * 10, -1, 1)

        return reward
