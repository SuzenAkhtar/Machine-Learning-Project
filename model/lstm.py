import torch.nn as nn


class YahooLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(YahooLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_size * 2,
                             num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        self.lstm2 = nn.LSTM(hidden_size * 4, hidden_size,
                             num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        self.ln = nn.LayerNorm(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.ln(out)
        out = out[:, -1, :]
        out = self.fc(out)

        return out
