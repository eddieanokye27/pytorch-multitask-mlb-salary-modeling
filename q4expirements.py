# Standalone file to test Question 4 experiments
import numpy
import torch
import matplotlib.pyplot as plt

def Question4(filepath):
    """
    Runs all Question 4 experiments on the MLB salary dataset.
    Returns:
        best_model: PyTorch model with best validation performance
        train_r2: R^2 on training set
        val_r2: R^2 on validation set
        test_r2: R^2 on held-out test set
    """
    #  Load and split dataset 
    data = numpy.loadtxt(filepath, delimiter=",", skiprows=1)
    x = data[:, 1:]
    y = data[:, 0]

    n = x.shape[0]
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    x_train = x[:train_size]
    y_train = y[:train_size]
    x_val = x[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    x_test = x[train_size+val_size:]
    y_test = y[train_size+val_size:]

    # Standardize features and target
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_std = (y_train - y_mean) / y_std

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_train_std = (x_train - x_mean) / x_std

    #  Helper functions 
    def create_mlp(input_dim, num_hidden_layers=3, neurons_per_layer=32, activation_fn=torch.nn.ReLU):
        layers = [torch.nn.LayerNorm(input_dim)]
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(in_dim, neurons_per_layer))
            layers.append(activation_fn())
            in_dim = neurons_per_layer
        layers.append(torch.nn.Linear(in_dim, 1))
        return torch.nn.Sequential(*layers)

    def train_mlp(x_train, y_train, model, lr=0.0005, weight_decay=1e-4, epochs=500):
        dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32),
                                                 torch.tensor(y_train, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.MSELoss()
        model.train()
        for _ in range(epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                y_pred = model(batch_X).squeeze()
                loss = loss_fn(y_pred, batch_y)
                loss.backward()
                optimizer.step()
        return model

    def evaluate_mlp(x, y, model, x_mean, x_std, y_mean, y_std):
        x_norm = (x - x_mean) / x_std
        model.eval()
        with torch.no_grad():
            pred_norm = model(torch.tensor(x_norm, dtype=torch.float32))
            pred = pred_norm.squeeze() * y_std + y_mean
        y_true = torch.tensor(y, dtype=torch.float32)
        ss_res = torch.sum((y_true - pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    #  Experiment 1: Number of neurons per hidden layer 
    neuron_options = [16, 32, 64, 128]
    r2_neurons = []
    for n in neuron_options:
        model = create_mlp(x_train.shape[1], neurons_per_layer=n)
        model = train_mlp(x_train_std, y_train_std, model, epochs=500)
        r2 = evaluate_mlp(x_val, y_val, model, x_mean, x_std, y_mean, y_std)
        r2_neurons.append(r2)
        print(f"Neurons {n}: R2={r2:.4f}")

    plt.figure()
    plt.plot(neuron_options, r2_neurons, marker='o')
    plt.xlabel("Neurons per Hidden Layer")
    plt.ylabel("Validation R^2")
    plt.title("Effect of Neuron Count")
    plt.show()

    #  Experiment 2: Number of hidden layers 
    layer_options = [1, 2, 3, 4]
    r2_layers = []
    for l in layer_options:
        model = create_mlp(x_train.shape[1], num_hidden_layers=l, neurons_per_layer=32)
        model = train_mlp(x_train_std, y_train_std, model, epochs=500)
        r2 = evaluate_mlp(x_val, y_val, model, x_mean, x_std, y_mean, y_std)
        r2_layers.append(r2)
        print(f"Hidden Layers {l}: R2={r2:.4f}")

    plt.figure()
    plt.plot(layer_options, r2_layers, marker='o')
    plt.xlabel("Number of Hidden Layers")
    plt.ylabel("Validation R^2")
    plt.title("Effect of Hidden Layer Count")
    plt.show()

    #  Experiment 3: Number of epochs 
    epoch_options = [100, 300, 500, 800]
    r2_epochs = []
    for e in epoch_options:
        model = create_mlp(x_train.shape[1], num_hidden_layers=3, neurons_per_layer=32)
        model = train_mlp(x_train_std, y_train_std, model, epochs=e)
        r2 = evaluate_mlp(x_val, y_val, model, x_mean, x_std, y_mean, y_std)
        r2_epochs.append(r2)
        print(f"Epochs {e}: R2={r2:.4f}")

    plt.figure()
    plt.plot(epoch_options, r2_epochs, marker='o')
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation R^2")
    plt.title("Effect of Number of Epochs")
    plt.show()

    #Experiment 4: Activation functions
    activation_options = [torch.nn.ReLU, torch.nn.Tanh, torch.nn.Sigmoid]
    r2_activations = []
    for act in activation_options:
        model = create_mlp(x_train.shape[1], num_hidden_layers=3, neurons_per_layer=32, activation_fn=act)
        model = train_mlp(x_train_std, y_train_std, model, epochs=500)
        r2 = evaluate_mlp(x_val, y_val, model, x_mean, x_std, y_mean, y_std)
        r2_activations.append(r2)
        print(f"Activation {act.__name__}: R2={r2:.4f}")

    plt.figure()
    plt.bar([act.__name__ for act in activation_options], r2_activations)
    plt.ylabel("Validation R^2")
    plt.title("Effect of Activation Function")
    plt.show()

    #  Select best model based on neurons experiment (example) 
    best_idx = r2_neurons.index(max(r2_neurons))
    best_model = create_mlp(x_train.shape[1], neurons_per_layer=neuron_options[best_idx])
    best_model = train_mlp(x_train_std, y_train_std, best_model, epochs=500)

    #  Final performance 
    train_r2 = evaluate_mlp(x_train, y_train, best_model, x_mean, x_std, y_mean, y_std)
    val_r2 = evaluate_mlp(x_val, y_val, best_model, x_mean, x_std, y_mean, y_std)
    test_r2 = evaluate_mlp(x_test, y_test, best_model, x_mean, x_std, y_mean, y_std)

    print(f"Best model performance -> Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}, Test R2: {test_r2:.4f}")

    return best_model, train_r2, val_r2, test_r2


#  Run experiments if this file is executed 
if __name__ == "__main__":
    best_model, train_r2, val_r2, test_r2 = Question4("./baseball.txt")
