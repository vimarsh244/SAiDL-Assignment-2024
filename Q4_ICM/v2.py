import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from collections import deque
import random
import numpy as np

# Define the state encoder (CNN)
class StateEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super(StateEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, hidden_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define the inverse model
class InverseModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(InverseModel, self).__init__()
        self.fc1 = nn.Linear(state_dim * 2 + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, next_state, action):
        x = torch.cat([state, next_state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the forward model
class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ForwardModel, self).__init__()
        self.state_encoder = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.prediction_head = nn.Linear(2 * hidden_dim, state_dim)

    def forward(self, states, action):
        state_encoding, _ = self.state_encoder(states)
        action_encoding = self.action_encoder(action)
        combined_encoding = torch.cat([state_encoding[:, -1], action_encoding], dim=1)
        next_state_prediction = self.prediction_head(combined_encoding)
        return next_state_prediction

# Define the DQN agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, buffer_size, batch_size, gamma, lr, update_freq, target_update_freq):
        self.state_encoder = StateEncoder(state_dim, hidden_dim)
        self.inverse_model = InverseModel(hidden_dim, action_dim, hidden_dim)
        self.forward_model = ForwardModel(hidden_dim, action_dim, hidden_dim)

        self.q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_network = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.inverse_optimizer = optim.Adam(self.inverse_model.parameters(), lr=self.lr)
        self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=self.lr)

        self.step_count = 0
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action = q_values.max(1)[1].item()
            return action

    def update(self):
        # Sample a batch from the replay buffer
        # if len(self.replay_buffer) >= self.batch_size:
        #     batch = random.sample(self.replay_buffer, self.batch_size)
        # else:
        #     print("Not enough items in replay buffer")
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Train the inverse model
        state_encodings = self.state_encoder(torch.stack([torch.from_numpy(state).float() for state in states]))
        next_state_encodings = self.state_encoder(torch.stack([torch.from_numpy(next_state).float() for next_state in next_states]))
        inverse_targets = torch.tensor(actions, dtype=torch.float32)
        inverse_predictions = self.inverse_model(state_encodings, next_state_encodings, inverse_targets)
        inverse_loss = nn.MSELoss()(inverse_predictions, inverse_targets)
        self.inverse_optimizer.zero_grad()
        inverse_loss.backward()
        self.inverse_optimizer.step()

        # Train the forward model
        state_sequences = torch.stack([self.state_encoder(torch.from_numpy(state).float().unsqueeze(0)) for state in states])
        next_state_encodings = self.state_encoder(torch.stack([torch.from_numpy(next_state).float() for next_state in next_states]))
        forward_targets = next_state_encodings
        forward_predictions = self.forward_model(state_sequences, torch.tensor(actions, dtype=torch.float32))
        forward_loss = nn.MSELoss()(forward_predictions, forward_targets)
        self.forward_optimizer.zero_grad()
        forward_loss.backward()
        self.forward_optimizer.step()

        # Train the Q-network
        q_targets = torch.tensor([reward + self.gamma * (1 - done) * self.target_q_network(torch.from_numpy(next_state).float().unsqueeze(0)).max().item() for reward, next_state, done in zip(rewards, next_states, dones)])
        q_predictions = self.q_network(torch.stack([torch.from_numpy(state).float() for state in states]))
        q_actions = torch.tensor([action for action in actions], dtype=torch.int64)
        q_loss = nn.MSELoss()(q_predictions.gather(1, q_actions.unsqueeze(1)), q_targets.unsqueeze(1))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update the target Q-network
        if self.step_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.step_count += 1
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train(self, env, num_episodes):
        
        # Exploration Phase
        state = env.reset()[0]
        for _ in range(100):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.replay_buffer.append((state, action, reward, next_state, done))
            state = next_state


        for episode in range(num_episodes):
            state = env.reset()[0]
            episode_reward = 0
            done = False

            while not done:
                # Get action using epsilon-greedy strategy
                action = self.get_action(state, self.epsilon)

                # Take action and observe next state and reward
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Store transition in the replay buffer
                self.replay_buffer.append((state, action, reward, next_state, done))

                # Train the models
                self.update()

                state = next_state
                episode_reward += reward

            print(f"Episode {episode}, Reward: {episode_reward}")

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim[0], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set up the environment
env = gym.make("Hopper-v2", render_mode="rgb_array")
state_dim = env.observation_space.shape
action_dim = env.action_space.shape[0]

# Initialize the DQN agent
agent = DQNAgent(state_dim, action_dim, hidden_dim=256, buffer_size=100000, batch_size=64, gamma=0.99, lr=1e-4, update_freq=4, target_update_freq=1000)

# Train the agent
agent.train(env, num_episodes=1000)