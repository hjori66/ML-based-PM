import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    """
    RNN Classifier
    """
    def __init__(self, device, n_dim=1, n_hidden=64, n_layer=1, n_label=2):
        super(RNNClassifier, self).__init__()
        self.n_layer = n_layer
        self.n_hidden = n_hidden
        self.device = device

        # self.embedding_layer = torch.nn.Linear(n_dim, n_hidden)
        self.rnn = nn.RNN(n_dim, n_hidden, n_layer, batch_first=True, nonlinearity='relu')
        self.lstm = nn.LSTM(n_dim, n_hidden, n_layer, batch_first=True, bidirectional=False)
        self.fc = torch.nn.Linear(n_hidden, n_label)

    def forward(self, x):
        h0 = torch.zeros(self.n_layer, x.size(0), self.n_hidden).to(device=self.device)
        # c0 = torch.zeros(self.n_layer, x.size(0), self.n_hidden).to(device=self.device)
        x, hn = self.rnn(x, h0)
        # x, (hn, cn) = self.lstm(x, (h0, c0))
        x = self.fc(x[:, -1, :])
        return x


class linearRegression(nn.Module):
    """
    일반적인 선형 모델
    """
    def __init__(self, slicer, n_dim=1, n_hidden=64, n_layer=1, n_label=2):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(slicer*n_dim, n_hidden)
        self.linear2 = torch.nn.Linear(n_hidden, n_hidden)
        self.linear3 = torch.nn.Linear(n_hidden, n_label)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x2 = self.linear(x)
        # x3 = self.linear2(x2)
        out = self.linear3(x2)
        return out
