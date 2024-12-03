# coding=utf-8

"""
Goal: Program Main.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import argparse

from tradingSimulator import TradingSimulator



###############################################################################
##################################### MAIN ####################################
###############################################################################

if __name__ == '__main__':
    # Retrieve the parameters sent by the user
    parser = argparse.ArgumentParser(description='Trading Simulator')
    parser.add_argument("-strategy", default='TDQN', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='Apple', type=str, help="Name of the stock (market)")
    parser.add_argument("-optimize", action='store_true', help="Run hyperparameter optimization")
    parser.add_argument("-n_trials", default=50, type=int, help="Number of trials for hyperparameter optimization")
    parser.add_argument("-rendering", action='store_true', help="Enable rendering during training and testing")
    args = parser.parse_args()
    
    # Initialization of the required variables
    simulator = TradingSimulator()
    strategy = args.strategy
    stock = args.stock
    optimize = args.optimize
    n_trials = args.n_trials
    rendering = args.rendering

    if optimize:
        # Run hyperparameter optimization
        simulator.optimizeHyperparameters(strategy, stock, n_trials=n_trials, rendering=rendering)
    else:
        # Training and testing of the trading strategy specified for the stock (market) specified
        simulator.simulateNewStrategy(strategy, stock, rendering=rendering, saveStrategy=False)
    """
    simulator.displayTestbench()
    simulator.analyseTimeSeries(stock)
    simulator.simulateNewStrategy(strategy, stock, saveStrategy=False)
    simulator.simulateExistingStrategy(strategy, stock)
    simulator.evaluateStrategy(strategy, saveStrategy=False)
    simulator.evaluateStock(stock)
    """
