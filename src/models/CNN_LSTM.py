import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, cnn : nn.Module):
        super().__init__()
        self.cnn = cnn

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            dropout=0.5,
            batch_first=True,
        )

        self.final_layer = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.final_layer(x)
        return x