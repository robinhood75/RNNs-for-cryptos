import torch
import torch.nn as nn

TIME_PERIOD = 1024


def train(model, train_set, train_targets, batch_size=16, nb_epochs=20):
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    assert batch_size*nb_epochs <= train_set.size(0), f"Train set not large enough for {nb_epochs} epochs"

    for epoch in range(nb_epochs):
        for b in range(0, train_set.size(0), batch_size):
            out = model.forward(train_set.narrow(0, b, batch_size))
            loss = loss(out, train_targets.narrow(0, b, batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Completed iteration {epoch+1}")

