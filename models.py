from dataset import load_object
import torch.nn as nn
import torch
import torch.nn.functional as F

class Basic_LSTM(nn.Module):

    def __init__(self, batch_size, lstm_num_hidden=64, lstm_num_layers=2,
                 input_dim=4, output_dim=10, device='cuda:0'):

        super(Basic_LSTM, self).__init__()

        self.lstm_num_hidden = lstm_num_hidden
        self.device = device

        #self.embed = nn.Embedding(vocabulary_size, embed_dim)
        self.lstm = nn.LSTM(input_dim, lstm_num_hidden, lstm_num_layers, bias=True)
        self.final = nn.Linear(lstm_num_hidden, output_dim)

    def forward(self, x):

        #x = self.embed(x)
        x, _ = self.lstm(x)
        x = self.final(x)

        return x






data = load_object('processed_data.pkl')