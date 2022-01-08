import torch.nn as nn


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size=64):
        super().__init__()
        self.obs_size = obs_size

        self.LSTM = nn.LSTM(input_size=obs_size, hidden_size=hidden_size)
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, obs):
        assert obs.size() == self.obs_size, f"Expected size {self.obs_size}, got {obs.size}"
        hidden_state = self.LSTM(obs)
        tmp = self.ReLU(self.fc1(hidden_state))
        return self.fc2(tmp)
