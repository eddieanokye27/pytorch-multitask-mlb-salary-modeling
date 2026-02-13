# Name this file assignment2.py when you submit
import numpy
import torch


# A function that implements a pytorch model following the provided description
class MultitaskNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Code for constructor goes here
        self.linear1 = torch.nn.Linear(3, 5)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(5, 4)
        self.activation2 = torch.nn.ReLU()
        self.output1 = torch.nn.Linear(4, 3)
        self.output2 = torch.nn.Linear(4, 3)
        # dim = 0 for along batch dim =  1 for class
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # Code for forward method goes here
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        y1 = self.output1(x)
        y2 = self.output2(x)
        y1 = self.output_activation(y1)
        y2 = self.output_activation(y2)
        return y1, y2


# A function that implements training following the provided description
def multitask_training(data_filepath: str) -> MultitaskNetwork:
    num_epochs = 100
    batch_size = 4

    data = numpy.loadtxt(data_filepath, delimiter=",")
    batches_per_epoch = int(data.shape[0] / batch_size)

    multitask_network = MultitaskNetwork()

    # Define loss function(s) here
    # compute categorical cross-entropy losses
    def compute_loss(
        y_a: torch.Tensor,
        y_b: torch.Tensor,
        y_pred_a: torch.Tensor,
        y_pred_b: torch.Tensor,
    ) -> torch.Tensor:
        # give credit to
        # https://discuss.pytorch.org/t/categorical-cross-entropy-loss-function-equivalent-in-pytorch/85165/10
        # https://discuss.pytorch.org/t/whats-different-between-dim-1-and-dim-0/61094
        # dim = 0 for along batch dim =  1 for class
        # https://discuss.pytorch.org/t/how-to-solve-the-loss-become-nan-because-of-using-torch-log/54499
        # sometime we might get log 0
        # for safe reason add a tiny value to avoid log 0 computation
        eps = 1e-7
        loss_a = -(y_a * torch.log(y_pred_a + eps)).sum(dim=1).mean()
        loss_b = -(y_b * torch.log(y_pred_b + eps)).sum(dim=1).mean()
        return loss_a + loss_b

    # Define optimizer here
    optimizer = torch.optim.SGD(multitask_network.parameters(), lr=0.001)
    # https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for _ in range(num_epochs):
        for batch_index in range(batches_per_epoch):
            x = torch.as_tensor(
                data[batch_index * batch_size : (batch_index + 1) * batch_size, 6:9],
                dtype=torch.float32,
            )
            y_a = torch.as_tensor(
                data[batch_index * batch_size : (batch_index + 1) * batch_size, 0:3],
                dtype=torch.float32,
            )
            y_b = torch.as_tensor(
                data[batch_index * batch_size : (batch_index + 1) * batch_size, 3:6],
                dtype=torch.float32,
            )

            y_pred_a, y_pred_b = multitask_network(x)

            # Compute loss here
            loss = compute_loss(y_a, y_b, y_pred_a, y_pred_b)

            # Compute gradients here
            optimizer.zero_grad()
            loss.backward()

            # Update parameters according to SGD with learning rate schedule here
            optimizer.step()
            scheduler.step()

    # A trained torch.nn.Module object
    return multitask_network


# A function that creates a pytorch model to predict the salary of an MLB position player
def mlb_position_player_salary(filepath):
    # filepath is the path to an csv file containing the dataset

    data = numpy.loadtxt(filepath, delimiter=",", skiprows=1)

    x = data[:, 1:]
    y = data[:, 0]

    train_size = int(0.8 * x.shape[0])
    x_train = x[:train_size, :]
    y_train = y[:train_size]
    x_val = x[train_size:, :]
    y_val = y[train_size:]

    # Preprocessing
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train = (y_train - y_mean) / y_std

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_train = (x_train - x_mean) / x_std

    class BaseballNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(16)
            self.layer1 = torch.nn.Linear(16, 32)
            self.relu = torch.nn.ReLU()
            self.layer2 = torch.nn.Linear(32, 64)
            self.layer3 = torch.nn.Linear(64, 32)
            self.layer4 = torch.nn.Linear(32, 1)

        def forward(self, x):
            x = self.norm(x)
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            x = self.layer3(x)
            x = self.relu(x)
            x = self.layer4(x)
            return x

    baseball_network = BaseballNetwork()

    loss_function = torch.nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_data = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    optimizer = torch.optim.Adam(baseball_network.parameters(), lr=0.0005, weight_decay=1e-4)

    for epoch in range(1150):
        for batch_X, batch_y in train_data:
            optimizer.zero_grad()
            y_train_pred = baseball_network(batch_X)
            loss = loss_function(torch.squeeze(y_train_pred), batch_y)
            loss.backward()
            optimizer.step()

    # Validation performance (R^2)
    x_val = (x_val - x_mean) / x_std

    baseball_network.eval()
    with torch.no_grad():
        pred_norm = baseball_network(torch.tensor(x_val, dtype=torch.float32))
        pred_val = pred_norm * y_std + y_mean

    y_true = torch.tensor(y_val, dtype=torch.float32)
    pred_val = torch.squeeze(pred_val)

    ss_res = torch.sum((y_true - pred_val) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    validation_performance = 1 - ss_res / ss_tot

    # model is a trained pytorch model for predicting the salary of an MLB position player
    # validation_performance is the performance of the model on a validation set
    return baseball_network, validation_performance


def main():

    print(multitask_training("./multitask_data.csv"))


if __name__ == "__main__":
    main()
