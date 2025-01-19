#!/usr/bin/env python3

# inference.py
# Usage: python inference.py
# Loads "my_best_ppo_model.pt" and tests it (deterministically) on a chosen date range.
import os
import json
from tradingSimulator import TradingSimulator

def main():
    # Create simulator instance
    simulator = TradingSimulator()

    stockSymbol = "AMZN" # or whichever stock
    subfolder = f"models/{stockSymbol}"
    
    # 1) Load best hyperparameters from best_params.json
    best_params_path = os.path.join(subfolder, "best_params.json")
    with open(best_params_path, "r") as f:
        PPO_PARAMS = json.load(f)
    # 2) Provide a default memory size if needed
    PPO_PARAMS.setdefault('MEMORY_SIZE', 10000)
    # 3) Run the saved model
    best_model_path = os.path.join(subfolder, "my_best_ppo_model.pt")

    # Run saved model with exact same parameters as in training
    simulator.runSavedModel(
        model_path=best_model_path,
        PPO_PARAMS=PPO_PARAMS,
        stockSymbol=stockSymbol,
        startingDate="2012-1-1", # training portion
        splitingDate="2018-1-1", # testing portion start
        endingDate="2020-1-1",
        observationSpace=30,
        actionSpace=2,
        money=100000,
        stateLength=30,
        transactionCosts=0.001,
        deterministic=True,
        rendering=True,
        showPerformance=True
    )

if __name__ == "__main__":
    main()