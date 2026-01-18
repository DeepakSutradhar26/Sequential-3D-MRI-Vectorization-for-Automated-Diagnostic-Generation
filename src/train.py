from torch.optim import Adam
import torch.nn as nn
import tqdm
import os

import architecture.cnn3 as arch
import models.CNN_LSTM as CNN_LSTM
import preprocess.normalization as pn
import metrics

def train_one_epoch():
    pass

def validate():
    pass

def main():
    pn.normalize_train()
    pn.normalize_test()

    cnn = arch.CNNArchitecture()
    model = CNN_LSTM(cnn).to(metrics.DEVICE)
    
    optimizer = Adam(model.parameters(), lr=metrics.LEARNING_RATE)

    for epoch in range(metrics.N_EPOCHS):
        pass

if __name__ == "__main__":
    main()