import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, obs_size, nb_currencies, hidden_size=32, *args):
        super().__init__()
        self.nb_currencies = nb_currencies
        self.obs_size = obs_size

        self.LSTM = nn.LSTM(input_size=(nb_currencies, obs_size), hidden_size=hidden_size)
        self.ReLU = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(size_2, nb_currencies)

    def forward(self, obs):
        raise NotImplementedError

