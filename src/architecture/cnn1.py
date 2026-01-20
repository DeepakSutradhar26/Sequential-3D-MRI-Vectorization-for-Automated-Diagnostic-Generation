import torch
import torch.nn as nn

class CNNArchitecture(nn.Module):
    def __init__(self, input_shape=(1, 128, 128, 32), dropout_rate=0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.conv_blocks = nn.Sequential(
            self.conv_block(1, 8),
            self.conv_block(8, 16),
            self.conv_block(16, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
        )

        _, D, H, W = input_shape
        for i in range(5):
            D, H, W = D // 2, H // 2, W // 2
        self.flat_dim = 128 * D * H * W

        self.cnn_input_size = 128

        self.final_layer = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.flat_dim, self.cnn_input_size),
            nn.BatchNorm1d(self.cnn_input_size),
            nn.ReLU(),
        )

    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout_rate),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = torch.flatten(x, start_dim=1) #(batch_size, 128, 1, 4, 4) -> (batch_size, 2048)
        x = self.final_layer(x) 
        return x 
