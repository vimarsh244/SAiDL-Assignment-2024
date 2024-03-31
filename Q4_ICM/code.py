import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np

# Define the environment
env = gym.make('Hopper-v2')
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Hyperparameters
lr = 1e-3
batch_size = 64
num_epochs = 100
num_frames = 4  # Number of frames to stack for the forward model

# Define Encoder (Linear)
class Encoder(nn.Module):
    def __init__(self, obs_dim, num_frames, latent_dim):
        super(Encoder, self).__init__()
        self.obs_dim = obs_dim
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.fc = nn.Linear(obs_dim * num_frames, latent_dim)

    def forward(self, x):
        x = x.view(-1, self.obs_dim * self.num_frames)
        return self.fc(x)

# Define Inverse Model (MLP)
class InverseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InverseModel, self).__init__()
        # print("input_dim:", input_dim)
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, s_enc, s_next_enc):
        # Ensure both inputs are reshaped to be 2-dimensional
        
        s_enc = s_enc.squeeze()
        s_next_enc = s_next_enc.squeeze()
        
        # print("s_enc shape:", s_enc.shape)
        # print("s_next_enc shape:", s_next_enc.shape)

        
        # Concatenate along dimension 1
        s_cat = torch.cat((s_enc, s_next_enc), dim=0)  
        # print("s_cat shape:", s_cat.shape)
        return self.fc(s_cat)

# Define Forward Model (MLP)
class ForwardModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ForwardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, s_enc, action):
        print("s_enc shape:", s_enc.shape)
        print("action shape:", action.shape)

        sa_cat = torch.cat((s_enc, action), dim=0)
        return self.fc(sa_cat)

# Initialize models
latent_dim = 128  # Dimension of latent space
encoder = Encoder(obs_dim, num_frames, latent_dim)
inverse_model = InverseModel(48, action_dim)
forward_model = ForwardModel(latent_dim + action_dim, latent_dim)

# Define optimizers
encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
inverse_optimizer = optim.Adam(inverse_model.parameters(), lr=lr)
forward_optimizer = optim.Adam(forward_model.parameters(), lr=lr)

# Loss functions
mse_loss = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    # Sample a batch of data
    obs_batch = []
    action_batch = []
    next_obs_batch = []
    for _ in range(batch_size):
        obs = env.reset()
        obs_buffer = np.zeros((num_frames, obs_dim))
        obs_buffer[0] = np.array(obs[0])
        for i in range(1, num_frames):
            action = env.action_space.sample()
            next_obs = env.step(action)
            # print(np.array(next_obs[0]))
            obs_buffer[i] = np.array(next_obs[0])
        obs_batch.append(obs_buffer[:-1])
        action_batch.append(action)
        next_obs_batch.append(obs_buffer[1:])

    obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
    action_batch = torch.tensor(action_batch, dtype=torch.float32)
    next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32)

    # Encode observations
    obs_enc = encoder(obs_batch)

    # Train Inverse Model
    pred_actions = inverse_model(obs_enc[:, 0], obs_enc[:, 1])
    inverse_loss = mse_loss(pred_actions, action_batch)

    inverse_optimizer.zero_grad()
    inverse_loss.backward()
    inverse_optimizer.step()

    # Train Forward Model
    pred_next_enc = forward_model(obs_enc[:, 0], action_batch)
    forward_loss = mse_loss(pred_next_enc, obs_enc[:, 1])

    forward_optimizer.zero_grad()
    forward_loss.backward()
    forward_optimizer.step()

    # Train Encoder
    pred_next_enc = forward_model(obs_enc[:, 0], action_batch)
    forward_loss = mse_loss(pred_next_enc, obs_enc[:, 1])

    encoder_optimizer.zero_grad()
    forward_loss.backward()
    encoder_optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Inverse Loss: {inverse_loss.item()}, Forward Loss: {forward_loss.item()}')
