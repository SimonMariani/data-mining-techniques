from dataset import load_object
import torch.nn as nn
import torch
import torch.nn.functional as F

class Basic_LSTM(nn.Module):

    def __init__(self, lstm_num_hidden=64, lstm_num_layers=2,
                 input_dim=38, output_dim=1):

        super(Basic_LSTM, self).__init__()

        self.lstm_num_hidden = lstm_num_hidden

        self.lstm = nn.LSTM(input_dim, lstm_num_hidden, lstm_num_layers, bias=True, batch_first=True)
        self.final = nn.Linear(lstm_num_hidden, output_dim)

    def forward(self, x):

        x, _ = self.lstm(x)
        x = self.final(x)

        return x

class Basic_BiLSTM(nn.Module):

    def __init__(self, lstm_num_hidden=64, lstm_num_layers=2,
                 input_dim=38, output_dim=1):

        super(Basic_BiLSTM, self).__init__()

        self.lstm_num_hidden = lstm_num_hidden

        self.lstm = nn.LSTM(input_dim, lstm_num_hidden, lstm_num_layers, bias=True, batch_first=True, bidirectional=True)
        self.final = nn.Linear(lstm_num_hidden*2, output_dim)

    def forward(self, x):

        x, _ = self.lstm(x)
        x = self.final(x)

        return x



class Basic_Net(nn.Module):

    def __init__(self, input_dim=38, output_dim=1):
        super(Basic_Net, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)

