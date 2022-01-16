import torch
import torch.nn as nn

# TODO


class TestUtils:
    def __init__(self, model, test_set, test_targets, discrete_targets=False, time_gap_test=1):
        """:param time_gap_test: number of iterations we want to predict ahead (default: 1)"""
        self.model = model
        self.test_set = test_set
        self.test_targets = test_targets
        self.discrete_targets = discrete_targets
        self.time_gap = time_gap_test

    def test(self):
        self.test_continuous_targets()

    def test_continuous_targets(self):
        """:returns: average MSE on the test set"""
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            predicted = self.model(self.test_set)
            loss = loss_fn(predicted, self.test_targets)
        print(f"\n Average MSE on {self.test_targets.size().numel()} test samples: {loss.tolist()}")

    def test_discrete_targets(self):
        """:returns: average probability of predicting the right class correctly"""
        pass
