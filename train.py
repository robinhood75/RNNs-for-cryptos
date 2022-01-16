import torch
import torch.nn as nn

TIME_PERIOD = 512


def train(model, train_set, train_targets, batch_size=16, nb_epochs=40):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    assert batch_size*nb_epochs <= train_set.size(0), f"Train set not large enough for {nb_epochs} epochs"

    for epoch in range(nb_epochs):
        out = model.forward(train_set.narrow(0, epoch*batch_size, batch_size))
        loss = loss_fn(out, train_targets.narrow(0, epoch*batch_size, batch_size))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Completed iteration {epoch+1}. Loss value: {loss}")

