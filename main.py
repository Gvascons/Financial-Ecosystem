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
import os
import json
from tradingSimulator import TradingSimulator

###############################################################################
##################################### MAIN ####################################
###############################################################################

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description='Trading Simulator - Training and Inference')
    
    # Required arguments
    parser.add_argument("-strategy", default='PPO', type=str, help="Name of the trading strategy")
    parser.add_argument("-stock", default='Apple', type=str, help="Name of the stock (market)")
    
    # Mode selection
    parser.add_argument("-mode", choices=['train', 'test', 'optimize'], default='train',
                      help="Mode to run: 'train' for training, 'test' for inference, 'optimize' for hyperparameter optimization")
    
    # Optional arguments
    parser.add_argument("-n_trials", default=50, type=int, help="Number of trials for hyperparameter optimization")
    parser.add_argument("-rendering", action='store_true', help="Enable rendering during training and testing")
    parser.add_argument("-model_path", type=str, help="Path to saved model for inference mode")
    parser.add_argument("-start_date", default="2012-1-1", type=str, help="Starting date for training/testing")
    parser.add_argument("-split_date", default="2024-1-1", type=str, help="Splitting date between train/test")
    parser.add_argument("-end_date", default="2025-1-1", type=str, help="Ending date for training/testing")
    parser.add_argument("-initial_money", default=100000, type=float, help="Initial capital")
    parser.add_argument("-transaction_costs", default=0.001, type=float, help="Transaction costs as fraction")
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = TradingSimulator()

    if args.mode == 'train':
        # Training mode
        simulator.simulateNewStrategy(
            args.strategy, 
            args.stock,
            startingDate=args.start_date,
            endingDate=args.end_date,
            splitingDate=args.split_date,
            money=args.initial_money,
            transactionCosts=args.transaction_costs,
            rendering=args.rendering,
            saveStrategy=True  # Always save model after training
        )

    elif args.mode == 'test':
        # Inference mode
        if args.strategy != 'PPO':
            print("Currently inference mode is only supported for PPO strategy")
            exit(1)
            
        # Construct paths based on stock symbol
        subfolder = f"models/{args.stock}"
        
        # Load hyperparameters
        best_params_path = os.path.join(subfolder, "best_params.json")
        if not os.path.exists(best_params_path):
            print(f"Error: Could not find hyperparameters at {best_params_path}")
            exit(1)
            
        with open(best_params_path, "r") as f:
            PPO_PARAMS = json.load(f)
        PPO_PARAMS.setdefault('MEMORY_SIZE', 10000)
        
        # Get model path
        model_path = args.model_path or os.path.join(subfolder, "my_best_ppo_model.pt")
        if not os.path.exists(model_path):
            print(f"Error: Could not find model at {model_path}")
            exit(1)
            
        # Run inference
        simulator.runSavedModel(
            model_path=model_path,
            PPO_PARAMS=PPO_PARAMS,
            stockSymbol=args.stock,
            startingDate=args.start_date,
            splitingDate=args.split_date,
            endingDate=args.end_date,
            observationSpace=30,
            actionSpace=2,
            money=args.initial_money,
            stateLength=30,
            transactionCosts=args.transaction_costs,
            deterministic=True,
            rendering=args.rendering,
            showPerformance=True
        )

    elif args.mode == 'optimize':
        # Hyperparameter optimization mode
        simulator.optimizeHyperparameters(
            args.strategy,
            args.stock,
            startingDate=args.start_date,
            endingDate=args.end_date,
            splitingDate=args.split_date,
            money=args.initial_money,
            transactionCosts=args.transaction_costs,
            n_trials=args.n_trials,
            rendering=args.rendering
        )
