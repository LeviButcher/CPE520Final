from math import ceil
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from pump_and_dump_dataset import PumpAndDumpDataset
from torch.utils.data import DataLoader


class RNN(nn.Module):
    def __init__(self, features, hidden_size, lstm_layers, time_series_size):
        super(RNN, self).__init__()
        self.features = features
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.time_series_size = time_series_size

        self.lstm = nn.LSTM(features, hidden_size, 1, batch_first=True)
        self.fully_connected = nn.Sequential(
            nn.Linear(self.time_series_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        o1, _ = self.lstm(x)
        n, _, _ = o1.shape

        x = o1.view(n, -1)
        return self.fully_connected(x)


def rnn_train_step(dataloader, model, loss_fn, optimizer):
    totalBatches = len(dataloader)
    totalLoss = 0
    for i, (X, Y) in enumerate(dataloader):
        # Compute prediction and loss

        pred = model(X).view(-1)
        loss = loss_fn(pred, Y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        totalLoss += loss
        if i % 100 == 0:
            print(f"loss: {loss:>7f} batch:{i}/{totalBatches}")

    return totalLoss / totalBatches


def evaluate_split_size(split_size, hidden_size, epochs, learning_rate, momentum):
    # Load Data
    dataset = PumpAndDumpDataset(split_size)
    n = len(dataset)
    trainSize = ceil(.6 * n)
    testSize = ceil(n - trainSize)
    trainSet, testSet = torch.utils.data.random_split(
        dataset, [trainSize, testSize], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(trainSet, batch_size=500, shuffle=True)
    test_dataloader = DataLoader(testSet, batch_size=500, shuffle=True)

    _, time_series_size, features = dataset.X.shape
    LSTM_Layers = 1

    # Setup Model
    model = RNN(features, hidden_size, LSTM_Layers, time_series_size).to()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum)

    # Run Training
    lossList = []
    for i in range(epochs):
        print(f"Epoch: {i}")
        loss = rnn_train_step(train_dataloader, model, loss_fn, optimizer)
        lossList.append(loss)

    # Plot Loss Curve
    plt.title("Training Loss")
    plt.plot(range(epochs), lossList)
    plt.savefig(f"images/training_loss_lstm-{split_size}.png")
    plt.close()

    # Evaluate Test Set
    pred, trueLabels = zip(*[(model(X).view(-1), Y)
                           for (X, Y) in test_dataloader])
    pred = torch.concat(pred)
    trueLabels = torch.concat(trueLabels)

    # Generate Confusion Matrix
    confusion_matrix = np.zeros((2, 2))

    for i in range(len(pred)):
        predicted = pred.round().int()[i]
        actual = trueLabels.int()[i]
        confusion_matrix[actual, predicted] += 1

    cmp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    fig, ax = plt.subplots(figsize=(10, 10))
    # cmp.plot(ax=ax, values_format=".5g")
    fig.savefig(f"images/lstm-{split_size}-confusion_matrix")
    plt.close()

    Y = trueLabels.detach().numpy()
    pred = pred.round().detach().numpy()

    accuracy = (pred == Y).sum() / len(Y) * 100
    precision = precision_score(Y, pred)
    recall = recall_score(Y, pred)
    f1 = f1_score(Y, pred)
    truePercentage = (Y >= 1).sum() / len(Y) * 100
    return np.array([split_size, truePercentage, accuracy, precision, recall, f1])


# Config / Model Setup
split_sizes = [5, 15, 25, 60]  # Measured in seconds
hidden_size = 1
epochs = 5
learning_rate = 1e-4
momentum = .9

res = [evaluate_split_size(s, hidden_size,
                           epochs, learning_rate, momentum) for s in split_sizes]
res_matrix = np.vstack(res)
np.savetxt("lstm_results.csv", res_matrix, delimiter=",")

print("End Program")
