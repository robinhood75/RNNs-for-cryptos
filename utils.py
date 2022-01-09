import numpy as np
import torch


def validate_time_period(data, time_period):
    assert data.size(1) >= time_period, "Time period should be smaller than data size"


def generate_train_and_test_sets(data, time_period, train_proportion=0.75, discrete_targets=False):
    """Shuffle data, extract train and tests i.e. a given number of consecutive entries

    :param data: size = (number of observed currencies, total time period of observation)
    :param time_period: number of consecutive entries given as an input to the model
    :param train_proportion: proportion of data allocated to train set, default 0.75
    :param discrete_targets: if set to True, targets are either class 0 (meaning current value < last value) or class 1
    (meaning current_value >= last_value). If set to False, target = (current_value - last_value)/last_value

    :returns: train set, test set, train_targets, test_targets"""
    validate_time_period(data, time_period=time_period)
    nb_samples = data.size(1) - time_period
    permutation = np.random.permutation(nb_samples)

    train_indices, test_indices = permutation[:int(train_proportion*nb_samples)], permutation[int(train_proportion*nb_samples):]

    train_set = torch.Tensor([[data[:, idx+t] for t in range(time_period)] for idx in train_indices])
    test_set = torch.Tensor([[data[:, idx + t] for t in range(time_period)] for idx in test_indices])

    if discrete_targets:
        raise NotImplementedError

    else:
        # TODO: document convention of putting predicted currency in 1st position of obs matrix
        train_targets = torch.Tensor([generate_target(data[0, idx + time_period - 1], data[0, idx + time_period])
                                      for idx in train_indices])
        test_targets = torch.Tensor([generate_target(data[0, idx + time_period - 1], data[0, idx + time_period])
                                     for idx in test_indices])

    return train_set, test_set, train_targets, test_targets


def generate_target(last_value, current_value, discrete_target=False):
    if discrete_target:
        return current_value >= last_value
    else:
        return (current_value - last_value)/last_value
