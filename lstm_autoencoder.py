from math import ceil, sqrt
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from pump_and_dump_dataset import PumpAndDumpDataset
from torch.utils.data import DataLoader


class LSTMAutoencoder(nn.Module):
    def __init__(self, features, hidden_size, lstm_layers, time_series_size):
        super(LSTMAutoencoder, self).__init__()
        self.features = features
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.time_series_size = time_series_size

        self.encode = nn.LSTM(
            input_size=features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.decode = nn.LSTM(
            input_size=hidden_size,
            hidden_size=features,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        e, _ = self.encode(x)
        d, _ = self.decode(e)
        return d


def rnn_train_step(dataloader, model, loss_fn, optimizer):
    totalBatches = len(dataloader)
    totalLoss = 0
    for i, (X, _) in enumerate(dataloader):
        # Compute prediction and loss

        pred = model(X)
        loss = loss_fn(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        totalLoss += loss
        if i % 100 == 0:
            print(f"loss: {loss:>7f} batch:{i}/{totalBatches}")

    return totalLoss / totalBatches


def predictPumpAndDump(X, model, loss_fn, threshold):
    n, _, _ = X.shape
    decX = model(X)
    pred = []

    for i in range(n):
        diff = loss_fn(decX[i], X[i])
        res = (diff.double() >= threshold).int().item()
        pred.append(res)

    return pred


def get_confusion_matrix(pred, Y):
    confusion_matrix = np.zeros((2, 2))

    for i in range(len(pred)):
        predicted = pred[i]
        actual = Y[i]
        confusion_matrix[actual][predicted] += 1

    return confusion_matrix


def compute_metrics(cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    total = tn + fp + fn + tp

    accuracy = tn + tp / total * 100
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    truePercentage = (tp + fn) / total * 100
    return np.array([truePercentage, accuracy, precision, recall, f1])


def find_best_threshold(val_dataloader, loss_fn, model):
    thresholds = [0.001, 0.0001]
    highestScore = 0
    bestThreshold = 0

    with torch.no_grad():
        for t in thresholds:
            for m in range(0, 100, 2):
                t = t + t * m
                pred, Y = zip(*[(predictPumpAndDump(X, model, loss_fn, t), Y.int().numpy())
                                for (X, Y) in val_dataloader])
                pred = np.concatenate(pred)
                Y = np.concatenate(Y)
                cm = get_confusion_matrix(pred, Y)
                _, _, _, _, f1 = compute_metrics(cm)

                if highestScore < f1:
                    bestThreshold = t
                    highestScore = f1

            # t = bestThreshold * m

            # pred, Y = zip(*[(predictPumpAndDump(X, model, loss_fn, t), Y.int().numpy())
            #                 for (X, Y) in val_dataloader])
            # pred = np.concatenate(pred)
            # Y = np.concatenate(Y)
            # cm = get_confusion_matrix(pred, Y)
            # _, _, _, _, f1 = compute_metrics(cm)

            # if highestScore < f1:
            #     bestThreshold = t

    return bestThreshold


def getLoss(loss_fn, X1, X2):
    res = [loss_fn(X1[i], X2[i]).item() for i in range(len(X1))]
    return np.array(res)


def evaluate_split_size(split_size, hidden_size, epochs, learning_rate, momentum, batch_size):
    # Load Data
    dataset = PumpAndDumpDataset(split_size)
    n = len(dataset)
    trainSize = ceil(.6 * n)
    valSize = ceil(.1 * n)
    testSize = ceil(n - (trainSize + valSize))

    trainSet, valSet, testSet = torch.utils.data.random_split(
        dataset, [trainSize, valSize, testSize], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(
        trainSet, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testSet, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valSet, batch_size=batch_size, shuffle=True)

    _, time_series_size, features = dataset.X.shape
    LSTM_Layers = 1

    # Setup Model
    model = LSTMAutoencoder(features, hidden_size,
                            LSTM_Layers, time_series_size).to()
    loss_fn = nn.L1Loss()
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
    plt.savefig(f"images/training_loss_lstm_encoder-{split_size}.png")
    plt.close()

    threshold = find_best_threshold(
        val_dataloader, loss_fn, model)

    # Evaluate Test Set
    pred, trueLabels = zip(*[(predictPumpAndDump(X, model, loss_fn, threshold), Y)
                             for (X, Y) in test_dataloader])
    pred = np.concatenate(pred).astype(int)
    Y = torch.concat(trueLabels).int().detach().numpy()

    # Generate Confusion Matrix
    confusion_matrix = get_confusion_matrix(pred, Y)

    cmp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    fig, ax = plt.subplots(figsize=(10, 10))
    cmp.plot(ax=ax, values_format=".5g")
    fig.savefig(f"images/lstm_encoder-{split_size}-confusion_matrix")
    plt.close()

    # accuracy = tn + tp / len(Y) * 100
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 * precision * recall / (precision + recall)
    # truePercentage = (Y >= 1).sum() / len(Y) * 100

    truePercentage, accuracy, precision, recall, f1 = compute_metrics(
        confusion_matrix)
    return np.array([threshold, split_size, truePercentage, accuracy, precision, recall, f1])


# Config / Model Setup
split_sizes = [60, 25, 15, 5]  # Measured in seconds
hidden_size = 1
epochs = 5
learning_rate = 1e-2
momentum = .9
batch_size = 100

res = [evaluate_split_size(s, hidden_size,
                           epochs, learning_rate, momentum, batch_size) for s in split_sizes]
res_matrix = np.vstack(res)
np.savetxt("lstm_encoder_results.csv", res_matrix, delimiter=",")


print("End Program")
