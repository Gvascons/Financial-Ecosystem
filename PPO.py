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
    """
    Neural network architecture for the PPO agent that combines LSTM and feedforward layers.
    
    The network processes sequential market data through an LSTM layer followed by
    shared feature extraction layers. It then splits into separate actor and critic
    heads for policy and value estimation respectively.

    Attributes:
        num_features (int): Number of input features per timestep
        sequence_length (int): Number of timesteps in the input sequence
        feature_dim (int): Dimension of the feature extraction layers
        lstm_hidden_size (int): Size of LSTM hidden states
        lstm_layers (int): Number of LSTM layers
        lstm (nn.LSTM): LSTM layer for sequential processing
        shared (nn.Sequential): Shared feature extraction layers
        actor (nn.Sequential): Policy head outputting action probabilities
        critic (nn.Sequential): Value head estimating state values
        hidden (tuple): LSTM hidden state cache (h_n, c_n)
    """
    def __init__(self, input_size, num_actions, PPO_PARAMS):
        super().__init__()
        self.PPO_PARAMS = PPO_PARAMS  # Store PPO_PARAMS as an instance variable
        
        # Set default values for missing parameters
        self.PPO_PARAMS.setdefault('LSTM_DROPOUT', 0.0)
        self.PPO_PARAMS.setdefault('LSTM_LAYERS', 2)
        self.PPO_PARAMS.setdefault('LSTM_HIDDEN_SIZE', 128)
        self.PPO_PARAMS.setdefault('HIDDEN_SIZE', 256)
        
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
    """
    Proximal Policy Optimization (PPO) agent for trading environments.
    
    This implementation uses a combined policy-value network architecture with
    LSTM layers to handle sequential market data. It includes features like
    Generalized Advantage Estimation (GAE) and clipped objective function.

    Attributes:
        device (torch.device): Device to run computations on (CPU/GPU)
        market_symbol (str): Trading symbol being trained on
        run_id (str): Unique identifier for the training run
        network (PPONetwork): Main neural network
        optimizer (torch.optim): Adam optimizer for network updates
        memory (deque): Replay buffer storing transitions
        training_step (int): Global step counter for training
        PPO_PARAMS (dict): Dictionary of hyperparameters including:
            - CLIP_EPSILON: PPO clipping parameter
            - VALUE_LOSS_COEF: Value function loss coefficient
            - ENTROPY_COEF: Entropy bonus coefficient
            - PPO_EPOCHS: Number of epochs to optimize on each batch
            - BATCH_SIZE: Size of training minibatches
            - GAMMA: Discount factor
            - GAE_LAMBDA: GAE parameter
            - LEARNING_RATE: Optimizer learning rate
            - MAX_GRAD_NORM: Gradient clipping threshold
            - HIDDEN_SIZE: Size of hidden layers
            - MEMORY_SIZE: Size of replay buffer
            - LSTM_HIDDEN_SIZE: Size of LSTM hidden states
            - LSTM_LAYERS: Number of LSTM layers
            - LSTM_DROPOUT: LSTM dropout probability
    """
    def __init__(self, state_dim, action_dim, PPO_PARAMS=None, device='cpu', marketSymbol=None, run_id=None):
        """Initialize PPO agent"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.market_symbol = marketSymbol
        self.run_id = run_id 
        
        print(f"Initialized PPO with run_id: {self.run_id}")

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
        
        # Ensure all required parameters exist with defaults if missing
        PPO_PARAMS.setdefault('MEMORY_SIZE', 10000)
        PPO_PARAMS.setdefault('LSTM_DROPOUT', 0.0)
        PPO_PARAMS.setdefault('LSTM_LAYERS', 2)
        PPO_PARAMS.setdefault('LSTM_HIDDEN_SIZE', 128)
        PPO_PARAMS.setdefault('HIDDEN_SIZE', 256)
        
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
        
        # Additional tracking variables
        self.prev_state = None

    def getNormalizationCoefficients(self, tradingEnv):
        """
        Calculate normalization coefficients for key features in the trading environment.

        This method scans through the environment's data (e.g., close prices, volumes, 
        plus any technical indicators) and computes min-max or specialized ranges 
        for each feature. These ranges will then be used by processState() to normalize 
        the data appropriately.

        Parameters:
        -----------
        tradingEnv : TradingEnv
            The trading environment instance containing the market data and 
            any precomputed technical indicators.

        Returns:
        --------
        coefficients : list of tuples
            A list of (min_val, max_val) tuples, one for each feature in the order 
            they will be processed in processState(). This allows for consistent 
            min-max or specialized normalization logic for each feature.
        """
        tradingData = tradingEnv.data
        coefficients = []
        margin = 1

        # 1. Close price => returns (absolute) => maximum value (absolute)
        closePrices = tradingData['Close'].tolist()
        returns = [abs((closePrices[i] - closePrices[i - 1]) / closePrices[i - 1]) for i in range(1, len(closePrices))]
        coeffs = (0, np.max(returns) * margin)
        coefficients.append(coeffs)

        # 2. Low/High prices => Delta prices => maximum value
        lowPrices = tradingData['Low'].tolist()
        highPrices = tradingData['High'].tolist()
        deltaPrice = [abs(highPrices[i] - lowPrices[i]) for i in range(len(lowPrices))]
        coeffs = (0, np.max(deltaPrice) * margin)
        coefficients.append(coeffs)

        # 3. Close/Low/High prices => Close price position => no normalization required
        coeffs = (0, 1)
        coefficients.append(coeffs)

        # 4. Volumes => minimum and maximum values
        volumes = tradingData['Volume'].tolist()
        coeffs = (np.min(volumes) / margin, np.max(volumes) * margin)
        coefficients.append(coeffs)

        # 5. Technical indicators
        technical_indicators = [
            'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
            'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'ATR_14', 'OBV'
        ]

        for indicator in technical_indicators:
            values = tradingData[indicator].tolist()
            if len(values) == 0:
                # Default if no values are available
                coeffs = (0, 1)
                coefficients.append(coeffs)
                continue

            # For RSI, which is typically 0 to 100
            if indicator == 'RSI_14':
                coeffs = (0, 100)  
            # For MACD-related indicators (can be negative)
            elif indicator in ['MACD', 'MACD_Signal', 'MACD_Hist']:
                max_abs = max(abs(np.min(values)), abs(np.max(values)))
                coeffs = (-max_abs * margin, max_abs * margin)
            else:
                # Generic min-max for other indicators
                coeffs = (np.min(values) / margin, np.max(values) * margin)

            coefficients.append(coeffs)

        return coefficients


    def processState(self, state, coefficients):
        """
        Convert raw environment state to a normalized format suitable for PPO.

        The PPO agent expects a consistent input shape across time steps. This method 
        takes the raw state (features + position) and applies normalization to ensure 
        stable network training.

        The first 17 entries represent distinct features derived from the environment:
        - Index 0 : Close prices (converted into returns).
        - Index 1 : Low prices (used in combination with High for delta).
        - Index 2 : High prices (used in combination with Low for delta).
        - Index 3 : Volumes.
        - Indices 4..16 : Technical indicators 
                            (e.g., SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV).
        The last entry (index 17) indicates the position dimension (−1 for short, 0 for no position, 
        +1 for long). This is intentionally excluded from normalization to preserve its meaning.

        Parameters:
        -----------
        state : list or ndarray
            The raw state as returned by the environment, shaped as [num_features], 
            where num_features = 18 in this setup (17 features + 1 position).
        coefficients : list of tuples
            A list of (min_val, max_val) tuples, specifying the range for each 
            feature to be used during normalization.

        Returns:
        --------
        processed_state : list or ndarray
            An updated copy of the input state with each feature normalized or otherwise 
            transformed as requested. The position dimension remains unchanged.

        ------------------------------

        Explanation for the elif i in [1, 2] logic:
            The code that calculates delta_price (High - Low) and the close_price_position ratio
            is purposefully placed in the branch where i == 1 (i.e., for Low).
            Even though we check i in [1,2], the actual work is only performed for i == 1
            because we need both Low (index 1) and High (index 2) at the same time.
            By the time the loop reaches i == 2, all the computations for High are already done,
            so we do nothing there.
        
        :param state: The original environment state (np.ndarray or similar).
        :param coefficients: A list of (min_val, max_val) tuples for each feature.
        :return: The state after applying feature-specific transformations and normalization.
        """
        processed_state = state.copy()

        for i in range(len(processed_state)):
            # If this is the position dimension, skip normalization
            if i == len(processed_state) - 1: # position dimension (-1, 0, or +1)
                continue

            feature = processed_state[i]
            min_val, max_val = coefficients[i]

            if i == 0:  # Close prices: convert to returns +normalizing
                returns = np.zeros_like(feature)
                returns[1:] = np.diff(feature) / feature[:-1]
                if min_val != max_val:
                    returns = np.clip((returns - min_val) / (max_val - min_val), -1, 1)
                processed_state[i] = returns

            elif i in [1, 2]:  # Low & High prices=> delta + close price position
                # We only handle Low & High at i == 1; i == 2 does nothing by design,
                # as everything needed for High is done in the same step
                if i == 1:
                    # 1. Compute delta price
                    delta_price = np.abs(processed_state[2] - processed_state[1])
                    if min_val != max_val:
                        delta_price = np.clip((delta_price - min_val) / (max_val - min_val), 0, 1)
                    processed_state[1] = delta_price

                    # 2. Derive the close_price_position ratio
                    close_price_returns = processed_state[0]
                    close_price_position = np.zeros_like(close_price_returns)
                    mask = delta_price != 0
                    close_price_position[mask] = (
                        np.abs(close_price_returns[mask] - processed_state[1][mask]) 
                        / delta_price[mask]
                    )
                    close_price_position[~mask] = 0.5

                    # 3. Optionally clip/rescale that ratio using the 3rd coefficient
                    min_val, max_val = coefficients[2]
                    if min_val != max_val:
                        close_price_position = np.clip(
                            (close_price_position - min_val) / (max_val - min_val),
                            0, 1
                        )
                    processed_state[2] = close_price_position

            else:  # Volume and Technical indicators
                if min_val != max_val:
                    processed_state[i] = np.clip(
                        (feature - min_val) / (max_val - min_val),
                        0, 1
                    )
                else:
                    processed_state[i] = np.zeros_like(feature)

        return processed_state

    def processReward(self, reward):
        """
        Same as in TDQN
        """
        rewardClipping = 1  # Assuming this is a global variable or define it here
        return np.clip(reward, -rewardClipping, rewardClipping)

    def select_action(self, state, deterministic=False):
        """
        Select an action from the current policy.

        Args:
            state (torch.Tensor or np.ndarray): Current environment state

        Returns:
            tuple: (action, log_prob, value) where:
                - action (int): Selected action index
                - log_prob (float): Log probability of selected action
                - value (float): Critic's value estimate for the state

        If deterministic=True,
        we pick argmax (the highest-probability action) instead of sampling
        from the distribution.
        """
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
            # We can use state.data_ptr() for a more efficient comparison
            if self.prev_state is None or state.data_ptr()!= self.prev_state.data_ptr():
                self.network.hidden = None
                self.prev_state = state  # No need to clone, juststore the reference
            
            with torch.no_grad():
                # Forward pass
                probs, value = self.network(state)
                dist = Categorical(probs)
                
                if deterministic:
                    # Pick the action with the highest probability
                    action = torch.argmax(probs, dim=-1)
                else:
                    # Sample stochastically from the distribution
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
        """
        Update policy and value networks using the PPO algorithm.
        
        This method:
        1. Computes advantages using GAE
        2. Normalizes advantages
        3. Performs multiple epochs of minibatch updates
        4. Uses clipped surrogate objective for policy updates
        5. Updates both actor and critic networks
        6. Applies gradient clipping
        7. Logs training metrics to TensorBoard
        """
        if len(self.memory) < self.PPO_PARAMS['BATCH_SIZE']:
            return
        
        # Ensure the network is in training mode
        self.network.train()
        # Convert stored transitions to tensors more efficiently
        # First convert lists to numpy arrays, then to tensors
        states = np.array([t['state'] for t in self.memory])
        states = torch.FloatTensor(states).to(self.device)
        
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

    def training(self, trainingEnv, trainingParameters=[], verbose=True, rendering=True, plotTraining=True, showPerformance=True):
        """
        Train the PPO agent on the given environment.

        Args:
            trainingEnv (TradingEnv): Environment to train on
            trainingParameters (list): List containing training parameters
            verbose (bool): Whether to print training progress
            rendering (bool): Whether to render training visualizations
            plotTraining (bool): Whether to plot training metrics
            showPerformance (bool): Whether to display performance metrics

        Returns:
            TradingEnv: Trained environment instance
        """
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

                    # Print feature names and values
                    """ print("Feature names and values:")
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
            
            # If required, show the rendering of the training environment
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
        """
        Test the trained policy on new market data.

        Args:
            trainingEnv (TradingEnv): Environment used for training (for normalization)
            testingEnv (TradingEnv): Environment to test on
            rendering (bool): Whether to render test visualizations
            showPerformance (bool): Whether to display performance metrics

        Returns:
            TradingEnv: Testing environment instance with results
        """
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
                    action, _, _ = self.select_action(state, deterministic=True)
                    
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

