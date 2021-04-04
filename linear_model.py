import torch
import numpy as np


class LinearModel(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=bias)
        torch.nn.init.kaiming_normal_(self.linear.weight)

        if bias:
            torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    x_train = np.linspace(0.0, 100.0, 100)
    y_train = 3 * x_train + 2

    model = LinearModel(1, 1)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10000):
        inputs = torch.from_numpy(np.expand_dims(x_train, axis=1)).float()
        labels = torch.from_numpy(np.expand_dims(y_train, axis=1)).float()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print("epoch: " + str(epoch) + ", loss: " + str(loss.item()))
