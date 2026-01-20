import torch
import torch.nn as nn

class CNNArchitecture(nn.Module):
    def __init__(self, input_shape=(1, 128, 128, 64)):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            self.conv_block(1, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 64),
            self.conv_block(64, 128),
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
    
    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return x 
