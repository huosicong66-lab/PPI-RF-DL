import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, config):
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0
        for e1, e2, label in dataloader:
            e1, e2, label = e1.to(config.device), e2.to(config.device), label.to(config.device).float()
            optimizer.zero_grad()
            output = model(e1, e2).squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
