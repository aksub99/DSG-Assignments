import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

import model


def generate_data(N):
    X = torch.randint(1, 1000, (N, 2))
    y = torch.zeros(N, )
    for i in range(N):
        y[i] = torch.sum(X[i])
    return X, y

criterion = nn.MSELoss()
optimizer = optim.SGD(model.fibnet.parameters(), lr=0.01)
X, y = generate_data(100)
fibnet = model.FibNet()
def train(n_epochs, N):
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i in range(N):
            inputs, labels = X[i], y[i]

            # zero the parameter gradients
            optimizer.zero_grad()
            # print(inputs)
            # forward + backward + optimize
            outputs = fibnet(inputs)
            # print(list(model.fibnet.parameters()))
            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 20 examples
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

train(100, 100)
