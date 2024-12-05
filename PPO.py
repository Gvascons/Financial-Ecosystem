# coding=utf-8

import math
import random
import copy
import datetime
import shutil
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import deque
from tradingPerformance import PerformanceEstimator
from dataAugmentation import DataAugmentation  # Make sure this is imported
from tradingEnv import TradingEnv  # Ensure this is available
import pandas as pd
import traceback

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# Create Figures directory if it doesn't exist
if not os.path.exists('Figs'):
    os.makedirs('Figs')

class PPONetwork(nn.Module):
    def __init__(self, input_size, num_actions, PPO_PARAMS):
        super().__init__()
        self.PPO_PARAMS = PPO_PARAMS  # Store PPO_PARAMS as an instance variable
        
        self.num_features = 18  # Total number of features
        self.sequence_length = 30
        self.feature_dim = self.PPO_PARAMS['HIDDEN_SIZE']
        self.lstm_hidden_size = self.PPO_PARAMS['LSTM_HIDDEN_SIZE']
        self.lstm_layers = self.PPO_PARAMS['LSTM_LAYERS']
        
        # LSTM expects input shape: [batch, sequence_length, num_features]
        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            dropout=self.PPO_PARAMS['LSTM_DROPOUT']
        )
        
        self.lstm_norm = nn.LayerNorm(self.lstm_hidden_size)
        
        # Rest of the network architecture remains the same
        self.shared = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LayerNorm(self.feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.LayerNorm(self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, num_actions)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(self.feature_dim // 2, self.feature_dim // 4),
            nn.LayerNorm(self.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 4, 1)
        )
        
        self._init_weights()
        self.hidden = None

    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)

    def init_hidden(self, batch_size, device):
        """Initialize LSTM hidden state"""
        return (torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(device),
                torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_size).to(device))

    def forward(self, x):
        assert x.is_cuda, "Input is not on CUDA"
        """
        Forward pass of the network.
        Expected input shape: [batch_size, num_features, sequence_length]
        """
        # Handle input preprocessing
        if isinstance(x, list):
            x = torch.FloatTensor(x).to(self.actor[0].weight.device)
        elif isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.actor[0].weight.device)
        
        # Add batch dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        # Reshape from [batch, features, sequence] to [batch, sequence, features]
        x = x.permute(0, 2, 1)
        
        batch_size = x.size(0)
        
        # Initialize hidden state if needed
        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            self.hidden = self.init_hidden(batch_size, x.device)
        
        # Process through LSTM
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        
        # Take the last output and normalize
        lstm_out = self.lstm_norm(lstm_out[:, -1, :])
        
        # Process through shared layers
        features = self.shared(lstm_out)
        
        # Get action probabilities and value
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits / 1.0, dim=-1)
        value = self.critic(features)
        
        return action_probs, value

class PPO:
    """Implementation of PPO algorithm for trading"""
    def __init__(self, state_dim, action_dim, PPO_PARAMS=None, device='cpu', marketSymbol=None, run_id=None):
        """Initialize PPO agent"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.market_symbol = marketSymbol
        self.run_id = run_id 
        
        # If PPO_PARAMS is None, use default parameters
        if PPO_PARAMS is None:
            PPO_PARAMS = {
                'CLIP_EPSILON': 0.2,
                'VALUE_LOSS_COEF': 0.5,
                'ENTROPY_COEF': 0.02,
                'PPO_EPOCHS': 4,
                'BATCH_SIZE': 128,
                'GAMMA': 0.99,
                'GAE_LAMBDA': 0.95,
                'LEARNING_RATE': 1e-4,
                'MAX_GRAD_NORM': 0.5,
                'HIDDEN_SIZE': 256,
                'MEMORY_SIZE': 10000,
                'LSTM_HIDDEN_SIZE': 128,
                'LSTM_LAYERS': 2,
                'LSTM_DROPOUT': 0.2,
            }
        self.PPO_PARAMS = PPO_PARAMS
        
        # Initialize network with correct input size
        self.input_size = state_dim
        self.num_actions = action_dim
        
        print(f"Initializing PPO with input size: {self.input_size}, action size: {self.num_actions}")
        
        self.network = PPONetwork(self.input_size, self.num_actions, self.PPO_PARAMS).to(self.device)
        print(f"Network device: {next(self.network.parameters()).device}")  # Debug print
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.PPO_PARAMS['LEARNING_RATE'])
        
        # Initialize memory
        self.memory = deque(maxlen=self.PPO_PARAMS['MEMORY_SIZE'])
        
        # Initialize training step counter
        self.training_step = 0
        
        # Initialize performance tracking
        self.best_reward = float('-inf')
        self.trailing_rewards = deque(maxlen=100)

        # Additional tracking variables
        self.market_symbol = marketSymbol
        self.prev_action = None
        self.prev_state = None

    def getNormalizationCoefficients(self, tradingEnv):
        """
        Same as in TDQN
        """
        # Retrieve the available trading data
        tradingData = tradingEnv.data
        closePrices = tradingData['Close'].tolist()
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        volumes = tradingData['Volume'].tolist()

        # Retrieve the coefficients required for the normalization
        coefficients = []
        margin = 1
        # 1. Close price => returns (absolute) => maximum value (absolute)
        returns = [abs((closePrices[i]-closePrices[i-1])/closePrices[i-1]) for i in range(1, len(closePrices))]
        coeffs = (0, np.max(returns)*margin)
        coefficients.append(coeffs)
        # 2. Low/High prices => Delta prices => maximum value
        deltaPrice = [abs(highPrices[i]-lowPrices[i]) for i in range(len(lowPrices))]
        coeffs = (0, np.max(deltaPrice)*margin)
        coefficients.append(coeffs)
        # 3. Close/Low/High prices => Close price position => no normalization required
        coeffs = (0, 1)
        coefficients.append(coeffs)
        # 4. Volumes => minimum and maximum values
        coeffs = (np.min(volumes)/margin, np.max(volumes)*margin)
        coefficients.append(coeffs)

        return coefficients

    def processState(self, state, coefficients):
        """
        Process the RL state returned by the environment
        (appropriate format and normalization)
        """
        # Create a copy of the state to avoid modifying the original
        processed_state = state.copy()
        
        # Process base features using original logic
        # Get the sequences for base features
        closePrices = processed_state[0]
        lowPrices = processed_state[1]
        highPrices = processed_state[2]
        volumes = processed_state[3]
        
        # 1. Close price => returns => MinMax normalization
        returns = np.zeros_like(closePrices)
        returns[1:] = np.diff(closePrices) / closePrices[:-1]
        if coefficients[0][0] != coefficients[0][1]:
            returns = np.clip((returns - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0]), -1, 1)
        processed_state[0] = returns
        
        # 2. Low/High prices => Delta prices => MinMax normalization
        deltaPrice = np.abs(highPrices - lowPrices)
        if coefficients[1][0] != coefficients[1][1]:
            deltaPrice = np.clip((deltaPrice - coefficients[1][0]) / (coefficients[1][1] - coefficients[1][0]), 0, 1)
        processed_state[1] = deltaPrice
        
        # 3. Close/Low/High prices => Close price position => No normalization required
        closePricePosition = np.zeros_like(closePrices)
        delta = np.abs(highPrices - lowPrices)
        mask = delta != 0
        closePricePosition[mask] = np.abs(closePrices[mask] - lowPrices[mask]) / delta[mask]
        closePricePosition[~mask] = 0.5
        if coefficients[2][0] != coefficients[2][1]:
            closePricePosition = np.clip((closePricePosition - coefficients[2][0]) / 
                                       (coefficients[2][1] - coefficients[2][0]), 0, 1)
        processed_state[2] = closePricePosition
        
        # 4. Volumes => MinMax normalization
        if coefficients[3][0] != coefficients[3][1]:
            volumes = np.clip((volumes - coefficients[3][0]) / (coefficients[3][1] - coefficients[3][0]), 0, 1)
        processed_state[3] = volumes
        
        # Process additional technical indicators (starting from index 4)
        for i in range(4, len(processed_state)):
            feature_data = processed_state[i]
            
            if i in [4, 5, 6, 7]:  # SMA and EMA features
                # Process like close prices (returns)
                returns = np.zeros_like(feature_data)
                returns[1:] = np.diff(feature_data) / feature_data[:-1]
                processed_state[i] = np.clip((returns - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0]), -1, 1)
                
            elif i == 8:  # RSI
                processed_state[i] = feature_data / 100.0
                
            elif i in [9, 10, 11]:  # MACD components
                mean = np.mean(feature_data)
                std = np.std(feature_data) + 1e-8
                processed_state[i] = np.clip((feature_data - mean) / std, -3, 3)
                
            elif i in [12, 13, 14]:  # Bollinger Bands
                if i == 12:  # Middle band - process like close prices
                    returns = np.zeros_like(feature_data)
                    returns[1:] = np.diff(feature_data) / feature_data[:-1]
                    processed_state[i] = np.clip((returns - coefficients[0][0]) / (coefficients[0][1] - coefficients[0][0]), -1, 1)
                else:  # Upper and Lower bands - relative to middle band
                    middle_band = processed_state[12]
                    processed_state[i] = (feature_data - middle_band) / (middle_band + 1e-8)
                    
            elif i == 15:  # ATR
                mean = np.mean(feature_data)
                std = np.std(feature_data) + 1e-8
                processed_state[i] = np.clip((feature_data - mean) / std, -3, 3)
                
            elif i == 16:  # OBV - process like volume
                processed_state[i] = np.clip((feature_data - coefficients[3][0]) / (coefficients[3][1] - coefficients[3][0]), 0, 1)
        
        return processed_state

    def processReward(self, reward):
        """
        Same as in TDQN
        """
        rewardClipping = 1  # Assuming this is a global variable or define it here
        return np.clip(reward, -rewardClipping, rewardClipping)

    def select_action(self, state):
        """Select an action from the current policy"""
        try:
            # Convert state to tensor and move to CUDA
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            elif isinstance(state, list):
                state = torch.FloatTensor(state)
            
            # Add batch dimension if needed
            if len(state.shape) == 2:
                state = state.unsqueeze(0)
            
            # Move to device
            state = state.to(self.device)
            
            """ # Add device verification
            if torch.cuda.is_available():
                print("\nGPU Verification in select_action:")
                print(f"Input state device: {state.device}")
                print(f"Network device: {next(self.network.parameters()).device}") """
            
            # Reset LSTM hidden state for new sequences
            if self.prev_state is None or not torch.equal(state, self.prev_state):
                self.network.hidden = None
            self.prev_state = state.clone()
            
            with torch.no_grad():
                probs, value = self.network(state)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
            
        except Exception as e:
            print(f"Error in select_action: {str(e)}")
            print(f"State type: {type(state)}")
            print(f"State shape: {np.shape(state) if isinstance(state, np.ndarray) else None}")
            raise

    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition in memory"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': float(reward),
            'next_state': next_state,
            'done': float(done),
            'log_prob': float(log_prob),
            'value': float(value)
        })

    def update_policy(self):
        """Update policy using PPO"""
        if len(self.memory) < self.PPO_PARAMS['BATCH_SIZE']:
            return
        
        # Ensure the network is in training mode
        self.network.train()

        # Add GPU memory monitoring
        """ if torch.cuda.is_available():
            print("\nGPU Memory Before Update:")
            print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
            print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f}MB") """

        # Convert stored transitions to tensors and ensure they're on CUDA
        states = np.array([t['state'] for t in self.memory])
        states = torch.FloatTensor(states).to(self.device)
        
        # Add debug print to verify device
        # print(f"Batch states device: {states.device}")  # Temporary debug line
        
        # Convert other data similarly
        actions = np.array([t['action'] for t in self.memory])
        actions = torch.LongTensor(actions).to(self.device)
        
        rewards = np.array([t['reward'] for t in self.memory])
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        next_states = np.array([t['next_state'] for t in self.memory])
        next_states = torch.FloatTensor(next_states).to(self.device)
        
        dones = np.array([t['done'] for t in self.memory])
        dones = torch.FloatTensor(dones).to(self.device)
        
        old_log_probs = np.array([t['log_prob'] for t in self.memory])
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        old_values = np.array([t['value'] for t in self.memory])
        old_values = torch.FloatTensor(old_values).to(self.device)
        
        # Rest of the method remains the same
        advantages = []
        gae = 0
        with torch.no_grad():
            for i in reversed(range(len(rewards))):
                next_value = 0 if i == len(rewards) - 1 else old_values[i + 1]
                delta = rewards[i] + self.PPO_PARAMS['GAMMA'] * next_value * (1 - dones[i]) - old_values[i]
                gae = delta + self.PPO_PARAMS['GAMMA'] * self.PPO_PARAMS['GAE_LAMBDA'] * (1 - dones[i]) * gae
                advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.PPO_PARAMS['PPO_EPOCHS']):
            # Sample mini-batches
            indices = np.random.permutation(len(self.memory))
            
            for start in range(0, len(self.memory), self.PPO_PARAMS['BATCH_SIZE']):
                end = start + self.PPO_PARAMS['BATCH_SIZE']
                batch_indices = indices[start:end]
                
                if len(batch_indices) < 3:
                    continue
                    
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Reset LSTM hidden state for each batch
                self.network.hidden = None
                
                # Get current policy outputs
                probs, values = self.network(batch_states)
                dist = Categorical(probs)
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Calculate losses separately
                ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.PPO_PARAMS['CLIP_EPSILON'], 1+self.PPO_PARAMS['CLIP_EPSILON']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), rewards[batch_indices])
                
                # Combine losses
                loss = (policy_loss + 
                       self.PPO_PARAMS['VALUE_LOSS_COEF'] * value_loss - 
                       self.PPO_PARAMS['ENTROPY_COEF'] * entropy)
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.PPO_PARAMS['MAX_GRAD_NORM'])
                self.optimizer.step()
                
                # Log metrics
                if hasattr(self, 'writer') and self.writer is not None:
                    self.writer.add_scalar('Loss/total', loss.item(), self.training_step)
                    self.writer.add_scalar('Loss/policy', policy_loss.item(), self.training_step)
                    self.writer.add_scalar('Loss/value', value_loss.item(), self.training_step)
                    self.writer.add_scalar('Loss/entropy', entropy.item(), self.training_step)
                
                self.training_step += 1
        
        # Clear memory after updates
        self.memory.clear()

        # After updates, check memory again
        """ if torch.cuda.is_available():
            print("\nGPU Memory After Update:")
            print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f}MB")
            print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f}MB") """

    def training(self, trainingEnv, trainingParameters=[], verbose=True, rendering=True, plotTraining=True, showPerformance=True):
        """Train the PPO agent"""
        try:
            num_episodes = trainingParameters[0] if trainingParameters else 1
            episode_rewards = []
            performanceTrain = []  # Track training performance
            performanceTest = []   # Track testing performance
            
            # Create run-specific directories and ID
            if self.run_id is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.run_id = f"run_PPO_{trainingEnv.marketSymbol}_{timestamp}"
            
            # Create base directories using run_id
            self.figures_dir = os.path.join('Figs', f'{self.run_id}')
            os.makedirs(self.figures_dir, exist_ok=True)
            os.makedirs('Results', exist_ok=True)
            
            # Initialize TensorBoard writer with the run_id
            self.writer = SummaryWriter(log_dir=f'runs/{self.run_id}')
            
            # Pass the directories to the training environment
            trainingEnv.figures_dir = self.figures_dir
            trainingEnv.results_dir = self.figures_dir
            
            # Apply data augmentation techniques to improve the training set
            dataAugmentation = DataAugmentation()
            trainingEnvList = dataAugmentation.generate(trainingEnv)
            
            # Initialize testing environment
            if plotTraining or showPerformance:
                marketSymbol = trainingEnv.marketSymbol
                startingDate = trainingEnv.endingDate
                endingDate = '2020-1-1'  # Adjust the ending date as needed
                money = trainingEnv.data['Money'][0]
                stateLength = trainingEnv.stateLength
                transactionCosts = trainingEnv.transactionCosts
                testingEnv = TradingEnv(marketSymbol, startingDate, endingDate, money, stateLength, transactionCosts)
                performanceTest = []
            
            # If required, print the training progression
            if verbose:
                print("Training progression (hardware selected => " + str(self.device) + "):")
            
            for episode in tqdm(range(num_episodes), disable=not(verbose)):
                # Reset action counts for this episode
                action_counts = {0: 0, 1: 0}
                
                # For each episode, train on the entire set of training environments
                for env_instance in trainingEnvList:
                    # Set the initial RL variables
                    coefficients = self.getNormalizationCoefficients(env_instance)
                    env_instance.reset()
                    startingPoint = random.randrange(len(env_instance.data.index))
                    env_instance.setStartingPoint(startingPoint)
                    state = self.processState(env_instance.state, coefficients)

                    """ # Print feature names and values
                    print("Feature names and values:")
                    for feature_name, feature_values in zip(trainingEnv.features, state):
                        print(f"{feature_name}: {feature_values}") """

                    done = False
                    steps = 0
                    
                    # Interact with the training environment until termination
                    while not done:
                        # Choose an action according to the RL policy and the current RL state
                        action, log_prob, value = self.select_action(state)
                        
                        # Track action counts
                        action_counts[action] = action_counts.get(action, 0) + 1
                        
                        # Interact with the environment with the chosen action
                        nextState, reward, done, info = env_instance.step(action)
                        
                        # Process the RL variables retrieved and store the experience
                        reward = self.processReward(reward)
                        nextState_processed = self.processState(nextState, coefficients)
                        self.store_transition(state, action, reward, nextState_processed, done, log_prob, value)
                        
                        # Execute the PPO learning procedure
                        if len(self.memory) >= self.PPO_PARAMS['BATCH_SIZE']:
                            self.update_policy()
                        
                        # Update the RL state
                        state = nextState_processed
                        steps += 1
                    
                    # Continuous tracking of the training performance
                    if plotTraining:
                        totalReward = sum([t['reward'] for t in self.memory])
                        episode_rewards.append(totalReward)
                
                # Compute both training and testing current performances
                if plotTraining or showPerformance:
                    # Training set performance
                    trainingEnv = self.testing(trainingEnv, trainingEnv, rendering=False, showPerformance=False)
                    analyser = PerformanceEstimator(trainingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTrain.append(performance)
                    self.writer.add_scalar('Training performance (Sharpe Ratio)', performance, episode)
                    trainingEnv.reset()
                    # Testing set performance
                    testingEnv = self.testing(trainingEnv, testingEnv, rendering=False, showPerformance=False)
                    analyser = PerformanceEstimator(testingEnv.data)
                    performance = analyser.computeSharpeRatio()
                    performanceTest.append(performance)
                    self.writer.add_scalar('Testing performance (Sharpe Ratio)', performance, episode)
                    testingEnv.reset()
                
                # Display action distribution at the end of each episode
                total_actions = sum(action_counts.values())
                print(f"\nAction Distribution during episode {episode}:")
                print(f"Short (0): {action_counts[0]} times ({(action_counts[0]/total_actions)*100:.1f}%)")
                print(f"Long (1): {action_counts[1]} times ({(action_counts[1]/total_actions)*100:.1f}%)")
            
            # Assess the algorithm performance on the training trading environment
            trainingEnv = self.testing(trainingEnv, trainingEnv)
            
            # If required, show the rendering of the trading environment
            if rendering:
                self.render_to_dir(trainingEnv)
            
            # If required, plot the training results
            if plotTraining:
                fig = plt.figure()
                ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
                ax.plot(performanceTrain)
                ax.plot(performanceTest)
                ax.legend(["Training", "Testing"])
                plt.savefig(os.path.join(self.figures_dir, f'TrainingTestingPerformance.png'))
                plt.close(fig)
                
                self.plotTraining(episode_rewards)
            
            # If required, print and save the strategy performance
            if showPerformance:
                analyser = PerformanceEstimator(trainingEnv.data)
                analyser.run_id = self.run_id  # Pass the full run_id
                analyser.displayPerformance('PPO', phase='training')
            
            return trainingEnv
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
        finally:
            if self.writer is not None:
                self.writer.flush()  # Ensure all pending events are written

    def testing(self, trainingEnv, testingEnv, rendering=True, showPerformance=True):
        """Test the trained policy on new data"""
        try:
            self.network.eval()
            coefficients = self.getNormalizationCoefficients(trainingEnv)
            state = testingEnv.reset()
            state = self.processState(state, coefficients)
            done = False
            episode_reward = 0  # Initialize episode_reward
            actions_taken = []
            action_counts = {0: 0, 1: 0}
            
            with torch.no_grad():
                while not done:
                    # Use the same sampling method as in training
                    action, _, _ = self.select_action(state)
                    
                    # Track action counts
                    action_counts[action] = action_counts.get(action, 0) + 1
                    
                    nextState, reward, done, _ = testingEnv.step(action)
                    state = self.processState(nextState, coefficients)
                    episode_reward += reward
                    actions_taken.append(action)
            
            # Display action distribution after testing
            total_actions = sum(action_counts.values())
            print("\nAction Distribution during testing:")
            print(f"Short (0): {action_counts[0]} times ({(action_counts[0]/total_actions)*100:.1f}%)")
            print(f"Long (1): {action_counts[1]} times ({(action_counts[1]/total_actions)*100:.1f}%)")
            
            # If required, show the rendering of the testing environment
            if rendering:
                self.render_to_dir(testingEnv)
            
            # If required, compute and display the strategy performance
            if showPerformance:
                analyser = PerformanceEstimator(testingEnv.data)
                analyser.run_id = self.run_id
                analyser.displayPerformance('PPO', phase='testing')
            
            return testingEnv
            
        except Exception as e:
            print(f"Error in testing: {str(e)}")
            raise

    def plotTraining(self, rewards):
        """Plot the training phase results (rewards)"""
        try:
            fig = plt.figure()
            ax1 = fig.add_subplot(111, ylabel='Total reward collected', xlabel='Episode')
            ax1.plot(rewards)
            plt.savefig(os.path.join(self.figures_dir, 'TrainingResults.png'))
            plt.close(fig)
        except Exception as e:
            print(f"Error in plotTraining: {str(e)}")

    def render_to_dir(self, env):
        """Render environment to run-specific directory"""
        try:
            env.render()
            base_dir = os.path.dirname(os.path.abspath(__file__))
            src_path = os.path.join(base_dir, 'Figs', f"{str(env.marketSymbol)}_Rendering.png")
            dst_path = os.path.join(self.figures_dir, f"{str(env.marketSymbol)}_Rendering.png")
            
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
        except Exception as e:
            print(f"Error in render_to_dir: {str(e)}")

    def __del__(self):
        """Cleanup method"""
        if hasattr(self, 'writer') and self.writer is not None:
            try:
                self.writer.close()
            except:
                pass

    def log_performance_metrics(self, episode, train_sharpe, test_sharpe):
        """Log performance metrics to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar('Performance/Train_Sharpe', train_sharpe, episode)
            self.writer.add_scalar('Performance/Test_Sharpe', test_sharpe, episode)
            
            # Log the difference between train and test Sharpe ratios to monitor overfitting
            self.writer.add_scalar('Performance/Train_Test_Gap', train_sharpe - test_sharpe, episode)

    def move_rendering_to_dir(self, env):
        """Move rendering file to the run-specific directory"""
        src_path = os.path.join('Figs', f'{str(env.marketSymbol)}_TrainingTestingRendering.png')
        dst_path = os.path.join(self.figures_dir, f'{str(env.marketSymbol)}_TrainingTestingRendering.png')
        
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

