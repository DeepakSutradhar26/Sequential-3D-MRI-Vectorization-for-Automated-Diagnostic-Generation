import torch
import torch.nn as nn
from torch.optim import NAdam
from tqdm import tqdm

import config
from models.CNN_LSTM import CNN_LSTM
from architecture.cnn1 import CNNArchitecture as CNN1
from architecture.cnn2 import CNNArchitecture as CNN2
from architecture.cnn3 import CNNArchitecture as CNN3
from data_pipeline.data_loader import train_loader, val_loader
from plots import plot_accuracy_curve, plot_all_accuracy, plot_all_loss, plot_loss_curve

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    epoch_loss = 0.0

    for x,y in tqdm(loader, desc="Training", leave=False):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
           preds = model(x)
           loss = criterion(preds, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        del preds, loss

    torch.cuda.empty_cache()
    return epoch_loss/len(loader) if len(loader) > 0 else 0

def validate(model, loader, criterion):
    model.eval()
    epoch_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x,y in tqdm(loader, desc="Validation", leave=False):
            x = x.to(config.DEVICE) #[8, 4, 1, 128, 128, 32]
            y = y.to(config.DEVICE) #[8, 1]

            preds_over_time = []

            preds = model(x)
            preds_over_time.append(preds)

            loss = criterion(preds, y)
            epoch_loss += loss.item()

            probs = torch.sigmoid(preds)
            preds = (probs > 0.5).float()

            correct += (preds == y).sum().item()
            total += y.size(0)
    
    acc = correct/total if total > 0 else 0
    return epoch_loss/len(loader) if len(loader) > 0 else 0, acc

def train_cnn_model(model, name, all_train_losses, all_val_losses, all_acc):
    optimizer = NAdam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []
    acc = []

    for epoch in range(config.N_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.N_EPOCHS}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        acc.append(val_acc)

        print(
            f"Train Loss : {train_loss:.2f} | Val Loss : {val_loss:.2f} | Accuracy : {val_acc:.2f}"
        )
    
    plot_loss_curve(train_losses, val_losses, name)
    plot_accuracy_curve(acc, name)
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_acc.append(acc)

    return optimizer

def reset_torch(model, optimizer):
    if model is not None:
        del model
    if optimizer is not None:
        del optimizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def main():
    all_train_losses = []
    all_val_losses = []
    all_acc = []

    model1 = CNN_LSTM(CNN1).to(config.DEVICE)
    op1 = train_cnn_model(model1, "CNN1", all_train_losses, all_val_losses, all_acc)
    reset_torch(model1, op1)

    model2 = CNN_LSTM(CNN2).to(config.DEVICE)
    op2 = train_cnn_model(model2, "CNN2", all_train_losses, all_val_losses, all_acc)
    reset_torch(model2, op2)

    model3 = CNN_LSTM(CNN3).to(config.DEVICE)
    op3 = train_cnn_model(model3, "CNN3", all_train_losses, all_val_losses, all_acc)
    reset_torch(model3, op3)

    plot_all_loss(all_train_losses, all_val_losses)
    plot_all_accuracy(all_acc)

if __name__ == "__main__":
    main()