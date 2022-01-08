import os
import torch

from observer.get_data import get_obs_from_json
from rnn.model import Net
from train import TIME_PERIOD, train

if __name__ == "__main__":
    # Load data
    obs_file = os.path.join("observer", "obs.json")
    data = get_obs_from_json(obs_file)

    obs_size = torch.Size([data.size(0), TIME_PERIOD])
    model = Net(obs_size=obs_size)

    train(model, data)
