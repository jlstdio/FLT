import json
import torch
from torch import nn


# Load Fisher information from a JSON file
def load_fisher(file_path, device):
    fisher = torch.load(file_path, map_location=device, weights_only=True)
    return fisher


# Save Fisher information to a file
def save_fisher(fisher, file_path):
    torch.save(fisher, file_path)


def visualize_fisher(fisher, file_path):
    pass


# Compute Fisher information for a client
def compute_fisher(model, dataloader, costFunc, device):
    fisher = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    model.to(device)
    model.eval()

    if costFunc == 'CEloss':
        criterion = nn.CrossEntropyLoss()
    elif costFunc == 'BCEloss':
        criterion = nn.BCELoss()
    elif costFunc == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()

    for inputs, targets in dataloader:
        inputs = inputs.to(device)

        if costFunc == 'CEloss':
            targets = targets.long().to(device)  # CE
        elif costFunc == 'BCEloss':
            targets = targets.to(device)  # BCE
        elif costFunc == 'BCEWithLogitsLoss':
            targets = targets.long().to(device)  # CE

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2)

    # Average Fisher information across batches
    fisher = {name: value / len(dataloader) for name, value in fisher.items()}
    return fisher


def compute_selective_fisher(model, dataloader, costFunc, device, significant_params):
    fisher = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    model.to(device)
    model.eval()

    if costFunc == 'CEloss':
        criterion = nn.CrossEntropyLoss()
    elif costFunc == 'BCEloss':
        criterion = nn.BCELoss()
    elif costFunc == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()

    for inputs, targets in dataloader:
        inputs = inputs.to(device)

        if costFunc == 'CEloss':
            targets = targets.long().to(device)  # CE
        elif costFunc == 'BCEloss':
            targets = targets.to(device)  # BCE
        elif costFunc == 'BCEWithLogitsLoss':
            targets = targets.long().to(device)  # CE

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2)

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2)

    # Average Fisher information across batches
    fisher = {name: value / len(dataloader) for name, value in fisher.items()}
    fisher = {n: fisher[n] for n in fisher if significant_params[n].sum() > 0}

    return fisher