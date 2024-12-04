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
from tradingEnv import TradingEnv  # Ensure this is available
import pandas as pd
import traceback

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create Figures directory if it doesn't exist
if not os.path.exists('Figs'):
    os.makedirs('Figs')

class PPONetwork(nn.Module):
    def __init__(self, input_size, num_actions, PPO_PARAMS):
        super().__init__()
        self.PPO_PARAMS = PPO_PARAMS
        print(f"Input size: {input_size}")
        self.num_features = input_size
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

    def forward(self, x, hidden):
        """
        Forward pass of the network.
        Expected input shape: [batch_size, sequence_length, num_features]
        """
        # Handle input preprocessing
        if isinstance(x, list):
            x = torch.FloatTensor(x).to(self.actor[0].weight.device)
        elif isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.actor[0].weight.device)

        # Add batch dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        batch_size = x.size(0)

        # Process through LSTM
        lstm_out, hidden = self.lstm(x, hidden)

        # Take the last output and normalize
        lstm_out = self.lstm_norm(lstm_out[:, -1, :])

        # Process through shared layers
        features = self.shared(lstm_out)

        # Get action probabilities and value
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(features)

        return action_probs, value, hidden

class PPO:
    """Implementation of PPO algorithm for trading"""
    def __init__(self, state_dim, action_dim, PPO_PARAMS=None, device=device, marketSymbol=None, run_id=None):
        """Initialize PPO agent"""
        self.device = device
        self.market_symbol = marketSymbol
        self.run_id = run_id 
        
        # Calculate actual input size based on features and sequence length
        self.sequence_length = 30  # This should match stateLength in TradingEnv
        self.num_features = state_dim  # This is the number of features per timestep
        
        print(f"Initialized PPO with run_id: {self.run_id}")
        
        # If PPO_PARAMS is None, use default parameters
        if PPO_PARAMS is None:
            PPO_PARAMS = {
                'CLIP_EPSILON': 0.2,
                'VALUE_LOSS_COEF': 0.5,
                'ENTROPY_COEF': 0.01,
                'PPO_EPOCHS': 10,
                'BATCH_SIZE': 64,
                'GAMMA': 0.99,
                'GAE_LAMBDA': 0.95,
                'LEARNING_RATE': 1e-4,
                'MAX_GRAD_NORM': 0.5,
                'HIDDEN_SIZE': 256,
                'LSTM_HIDDEN_SIZE': 128,
                'LSTM_LAYERS': 2,
                'LSTM_DROPOUT': 0.2,
                'TIMESTEPS_PER_BATCH': 2048,
                'MINI_BATCH_SIZE': 64,
            }
        self.PPO_PARAMS = PPO_PARAMS

        # Initialize network with correct input size
        self.input_size = state_dim  # This should be the number of features per timestep
        self.num_actions = action_dim

        print(f"Initializing PPO with input size: {self.input_size}, action size: {self.num_actions}")

        self.network = PPONetwork(self.input_size, self.num_actions, self.PPO_PARAMS).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.PPO_PARAMS['LEARNING_RATE'])

        # Initialize memory
        self.memory = []

        # Initialize training step counter
        self.training_step = 0

        # Initialize performance tracking
        self.best_reward = float('-inf')
        self.trailing_rewards = deque(maxlen=100)

        # Additional tracking variables
        self.market_symbol = marketSymbol

        # Add flag for state printing
        self._has_printed_state = False

    def getNormalizationCoefficients(self, tradingEnv):
        """
        Compute normalization coefficients for input features
        """
        tradingData = tradingEnv.data
        coefficients = {}
        for feature in tradingEnv.features + ['Position']:
            mean = tradingData[feature].mean()
            std = tradingData[feature].std()
            coefficients[feature] = (mean, std)
        return coefficients

    def processState(self, state, coefficients, features):
        """
        Normalize the RL state using z-score normalization
        Returns shape: [sequence_length, num_features]
        """
        processed_state = []
        num_features = len(features)
        
        # Reshape state if necessary
        if isinstance(state, np.ndarray) and state.shape[0] == num_features:
            state = state.T  # Transpose to get [sequence_length, features]
        
        for feature_idx in range(num_features):
            feature_name = features[feature_idx]
            feature_data = state[:, feature_idx] if len(state.shape) > 1 else state[feature_idx]
            
            mean, std = coefficients.get(feature_name, (0, 1))
            if std == 0:
                std = 1
            normalized_data = (feature_data - mean) / std
            processed_state.append(normalized_data)
        
        processed_state = np.array(processed_state, dtype=np.float32).T
        return processed_state  # Shape: [sequence_length, num_features]

    def processReward(self, reward):
        """
        Optionally process the reward
        """
        return reward

    def select_action(self, state, hidden):
        """
        Select an action from the current policy
        state shape expected: [sequence_length, num_features]
        """
        # Print state information only once per agent
        if not self._has_printed_state:
            print("\nState shape:", state.shape)
            if len(state.shape) == 2:
                seq_len, num_features = state.shape
                print(f"Sequence length: {seq_len}, Number of features: {num_features}")
                
                # Create a readable format of the last state
                last_state = state[-1]  # Get the most recent state
                feature_names = self.network.features if hasattr(self.network, 'features') else [
                    'Open', 'High', 'Low', 'Close', 'Volume',
                    'SMA_10', 'SMA_20', 'EMA_10', 'EMA_20',
                    'RSI_14',
                    'MACD', 'MACD_Signal', 'MACD_Hist',
                    'BB_Middle', 'BB_Upper', 'BB_Lower',
                    'ATR_14', 'OBV',
                    'Position'
                ]
                
                print("\nMost recent state values:")
                for i, value in enumerate(last_state):
                    feature_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                    print(f"{feature_name}: {value:.4f}")
                
                self._has_printed_state = True
        
        # Convert to tensor and add batch dimension
        state = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)  # Add batch dimension [1, sequence_length, num_features]
        
        # Verify dimensions
        batch_size, seq_len, num_features = state.shape
        if num_features != self.input_size:  # Changed from self.num_features to self.input_size
            print(f"State shape: {state.shape}")
            print(f"Expected features: {self.input_size}")  # Changed from self.num_features
            print(f"Got features: {num_features}")
            raise ValueError(f"Feature dimension mismatch. Expected {self.input_size}, got {num_features}")
        
        action_probs, value, hidden = self.network(state, hidden)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item(), hidden

    def store_transition(self, transition):
        """Store a transition in memory"""
        self.memory.append(transition)

    def update_policy(self):
        """Update policy using PPO"""
        if len(self.memory) == 0:
            return

        # Set network to training mode
        self.network.train()
        
        # Convert stored transitions to tensors
        states = [t['state'] for t in self.memory]
        actions = torch.tensor([t['action'] for t in self.memory], dtype=torch.long, device=self.device)
        rewards = [t['reward'] for t in self.memory]
        dones = [t['done'] for t in self.memory]
        log_probs = torch.tensor([t['log_prob'] for t in self.memory], dtype=torch.float32, device=self.device)
        values = torch.tensor([t['value'] for t in self.memory], dtype=torch.float32, device=self.device)

        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0
        next_value = 0
        for i in reversed(range(len(rewards))):
            mask = 1.0 - dones[i]
            delta = rewards[i] + self.PPO_PARAMS['GAMMA'] * next_value * mask - values[i]
            gae = delta + self.PPO_PARAMS['GAMMA'] * self.PPO_PARAMS['GAE_LAMBDA'] * mask * gae
            advantages.insert(0, gae)
            next_value = values[i]
            returns.insert(0, gae + values[i])

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare data for minibatch updates
        dataset = list(zip(states, actions, returns, advantages, log_probs))
        batch_size = self.PPO_PARAMS['MINI_BATCH_SIZE']
        for _ in range(self.PPO_PARAMS['PPO_EPOCHS']):
            random.shuffle(dataset)
            for i in range(0, len(dataset), batch_size):
                minibatch = dataset[i:i+batch_size]
                batch_states = [mb[0] for mb in minibatch]
                
                # Fix tensor construction warnings by using clone().detach()
                batch_actions = torch.stack([mb[1].clone().detach() if isinstance(mb[1], torch.Tensor) 
                                           else torch.tensor(mb[1], dtype=torch.long) 
                                           for mb in minibatch]).to(self.device)
                
                batch_returns = torch.stack([mb[2].clone().detach() if isinstance(mb[2], torch.Tensor)
                                           else torch.tensor(mb[2]) 
                                           for mb in minibatch]).to(self.device)
                
                batch_advantages = torch.stack([mb[3].clone().detach() if isinstance(mb[3], torch.Tensor)
                                              else torch.tensor(mb[3]) 
                                              for mb in minibatch]).to(self.device)
                
                batch_old_log_probs = torch.stack([mb[4].clone().detach() if isinstance(mb[4], torch.Tensor)
                                                  else torch.tensor(mb[4]) 
                                                  for mb in minibatch]).to(self.device)

                # Pad sequences and pack them
                batch_lengths = [len(s) for s in batch_states]
                padded_states = nn.utils.rnn.pad_sequence([torch.tensor(s) for s in batch_states], batch_first=True)
                packed_states = nn.utils.rnn.pack_padded_sequence(padded_states, batch_lengths, batch_first=True, enforce_sorted=False)

                # Initialize hidden state
                batch_size_mb = len(minibatch)
                h_0 = torch.zeros(self.network.lstm_layers, batch_size_mb, self.network.lstm_hidden_size).to(self.device)
                c_0 = torch.zeros(self.network.lstm_layers, batch_size_mb, self.network.lstm_hidden_size).to(self.device)
                hidden = (h_0, c_0)

                # Forward pass
                action_probs, values, _ = self.network(padded_states.to(self.device), hidden)
                dist = Categorical(action_probs)
                curr_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Calculate losses
                ratios = torch.exp(curr_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.PPO_PARAMS['CLIP_EPSILON'], 1 + self.PPO_PARAMS['CLIP_EPSILON']) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.view(-1), batch_returns.view(-1))
                loss = policy_loss + self.PPO_PARAMS['VALUE_LOSS_COEF'] * value_loss - self.PPO_PARAMS['ENTROPY_COEF'] * entropy

                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.PPO_PARAMS['MAX_GRAD_NORM'])
                self.optimizer.step()

        # Clear memory after updates
        self.memory = []
        # Set back to eval mode
        self.network.eval()

    def training(self, trainingEnv, trainingParameters=[], verbose=True, rendering=True, plotTraining=True, showPerformance=True):
        """Train the PPO agent"""
        try:
            num_episodes = trainingParameters[0] if trainingParameters else 1
            episode_rewards = []
            performanceTrain = []
            performanceTest = []
            total_timesteps = num_episodes * self.PPO_PARAMS['TIMESTEPS_PER_BATCH']
            timestep = 0



            if verbose:
                print(f"Training on device: {self.device}")
                print(f"Number of episodes: {num_episodes}")
                print("Starting training...")

            if self.run_id is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.run_id = f"run_PPO_{trainingEnv.marketSymbol}_{timestamp}"

            # Create directories
            self.figures_dir = os.path.join('Figs', f'{self.run_id}')
            os.makedirs(self.figures_dir, exist_ok=True)
            os.makedirs('Results', exist_ok=True)

            # Initialize TensorBoard writer
            self.writer = SummaryWriter(log_dir=f'runs/{self.run_id}')

            # Pass the directories to the training environment
            trainingEnv.figures_dir = self.figures_dir
            trainingEnv.results_dir = self.figures_dir

            if verbose:
                print("Training progression (device selected => " + str(self.device) + "):")

            # Get normalization coefficients and features
            coefficients = self.getNormalizationCoefficients(trainingEnv)
            features = trainingEnv.features + ['Position']

            timestep = 0
            progress_bar = tqdm(range(num_episodes), desc='Training Progress', 
                               position=1, leave=True, disable=not verbose)
            
            with tqdm(total=num_episodes, desc="Episodes", position=1, leave=True) as episode_bar, \
             tqdm(total=total_timesteps, desc="Timesteps", position=2, leave=True) as timestep_bar:
                for episode in progress_bar:
                    state = trainingEnv.reset()
                    state = self.processState(state, coefficients, features)
                    done = False
                    episode_reward = 0
                    hidden = self.network.init_hidden(1, self.device)

                    while not done:
                        action, log_prob, value, hidden = self.select_action(state, hidden)
                        next_state, reward, done, _ = trainingEnv.step(action)
                        next_state_processed = self.processState(next_state, coefficients, features)
                        transition = {
                            'state': state,
                            'action': action,
                            'reward': reward,
                            'done': done,
                            'log_prob': log_prob,
                            'value': value
                        }
                        self.store_transition(transition)

                        state = next_state_processed
                        episode_reward += reward
                        timestep += 1
                        timestep_bar.update(1)  # Update timestep progress

                        if timestep % self.PPO_PARAMS['TIMESTEPS_PER_BATCH'] == 0:
                            self.update_policy()

                    # Update policy at the end of each episode
                    self.update_policy()
                    episode_bar.update(1)  # Update episode progress

                    episode_rewards.append(episode_reward)
                    self.writer.add_scalar('Episode Reward', episode_reward, episode)

                    if plotTraining or showPerformance:
                        # Evaluate performance on training data
                        trainingEnv_eval = self.testing(trainingEnv, trainingEnv, rendering=False, showPerformance=False)
                        analyser = PerformanceEstimator(trainingEnv_eval.data)
                        performance = analyser.computeSharpeRatio()
                        performanceTrain.append(performance)
                        self.writer.add_scalar('Training performance (Sharpe Ratio)', performance, episode)
                        trainingEnv_eval.reset()

                    # Update progress bar with current reward
                    progress_bar.set_postfix({'Reward': f'{episode_reward:.2f}'})

            if rendering:
                self.render_to_dir(trainingEnv)

            if plotTraining:
                fig = plt.figure()
                ax = fig.add_subplot(111, ylabel='Performance (Sharpe Ratio)', xlabel='Episode')
                ax.plot(performanceTrain)
                ax.legend(["Training"])
                plt.savefig(os.path.join(self.figures_dir, f'TrainingPerformance.png'))
                plt.close(fig)

                self.plotTraining(episode_rewards)

            if showPerformance:
                analyser = PerformanceEstimator(trainingEnv.data)
                analyser.run_id = self.run_id
                analyser.displayPerformance('PPO', phase='training')

            return trainingEnv

        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
        finally:
            if self.writer is not None:
                self.writer.flush()

    def testing(self, trainingEnv, testingEnv, rendering=True, showPerformance=True):
        """Test the trained policy on new data"""
        try:
            self.network.eval()
            coefficients = self.getNormalizationCoefficients(trainingEnv)
            features = trainingEnv.features + ['Position']
            state = testingEnv.reset()
            state = self.processState(state, coefficients, features)
            done = False
            episode_reward = 0
            actions_taken = []
            action_counts = {0: 0, 1: 0, 2: 0}
            hidden = self.network.init_hidden(1, self.device)

            with torch.no_grad():
                while not done:
                    action, _, _, hidden = self.select_action(state, hidden)
                    next_state, reward, done, _ = testingEnv.step(action)
                    state = self.processState(next_state, coefficients, features)  # Pass features parameter
                    episode_reward += reward
                    actions_taken.append(action)
                    action_counts[action] = action_counts.get(action, 0) + 1

            # Display action distribution after testing
            total_actions = sum(action_counts.values())
            print("\nAction Distribution during testing:")
            for action, count in action_counts.items():
                print(f"Action {action}: {count} times ({(count/total_actions)*100:.1f}%)")

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

    def loadModel(self, fileName):
        """Load model parameters"""
        self.network.load_state_dict(torch.load(fileName))

    def saveModel(self, fileName):
        """Save model parameters"""
        torch.save(self.network.state_dict(), fileName)
