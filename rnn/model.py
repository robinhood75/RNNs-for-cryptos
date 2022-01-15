import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, obs_shape, hidden_size=8, fc_size=64):
        """:param obs_shape: tuple (time period, nb features observed at time t)"""
        super().__init__()
        self.obs_shape = obs_shape
        time_period, n_features = obs_shape[0], obs_shape[1]

        self.LSTM = nn.LSTM(input_size=time_period, hidden_size=hidden_size, batch_first=True)
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(n_features*hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, 1)

    def forward(self, obs):
        assert obs.shape[-2:] == self.obs_shape, f"Expected size {self.obs_shape}, got {obs.shape}"
        obs = obs.permute(0, 2, 1)
        hidden_state = self.LSTM(obs)[0]
        hidden_state_flattened = torch.flatten(hidden_state, start_dim=1)
        tmp = self.ReLU(self.fc1(hidden_state_flattened))
        return self.fc2(tmp)
