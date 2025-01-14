#!/usr/bin/env python3

# inference.py
# Usage: python inference.py
# Loads "my_best_ppo_model.pt" and tests it (deterministically) on a chosen date range.

import json
from tradingSimulator import TradingSimulator

def main():
    # Create simulator instance
    simulator = TradingSimulator()

    # Load best hyperparameters from best_params.json
    with open("best_params.json", "r") as f:
        PPO_PARAMS = json.load(f)
    
    # Ensure MEMORY_SIZE exists
    PPO_PARAMS.setdefault('MEMORY_SIZE', 10000)

    # Run saved model with exact same parameters as in training
    simulator.runSavedModel(
        model_path="my_best_ppo_model.pt",
        PPO_PARAMS=PPO_PARAMS,
        stockSymbol="AMZN",  # Make sure this matches your training stock
        # Use the same date ranges from your final testing phase
        startingDate="2012-1-1",   # Your test start date
        splitingDate="2018-1-1",   # Your training start date (for normalization)
        endingDate="2020-1-1",     # Your test end date
        observationSpace=30,        # Must match your training setup
        actionSpace=2,
        money=100000,              # Same as training
        stateLength=30,            # Same as training
        transactionCosts=0.001,    # Same as training
        deterministic=True,        # For reproducible results
        rendering=True,
        showPerformance=True
    )

if __name__ == "__main__":
    main()