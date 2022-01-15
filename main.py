import os
import torch

from observer.get_data import get_obs_from_json
from rnn.model import Net
from train import TIME_PERIOD, train
from utils import generate_train_and_test_sets

if __name__ == "__main__":
    # Load data
    obs_file = os.path.join("observer", "obs.json")
    data = get_obs_from_json(obs_file).permute(1, 0)

    # Create model
    obs_size = torch.Size([TIME_PERIOD, data.size(1)])
    model = Net(obs_shape=obs_size)

    train_set, test_set, train_targets, test_targets = generate_train_and_test_sets(data, TIME_PERIOD)
    train(model, train_set, train_targets)
