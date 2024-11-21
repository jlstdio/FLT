import json
import torch


# Load Fisher information from a JSON file
def load_fisher_from_json(file_path):
    with open(file_path, 'r') as f:
        fisher_json = json.load(f)
    fisher = {name: torch.tensor(param) for name, param in fisher_json.items()}
    return fisher


# Save Fisher information to a JSON file
def save_fisher_to_json(fisher, file_path):
    fisher_json = {name: param.tolist() for name, param in fisher.items()}
    with open(file_path, 'w') as f:
        json.dump(fisher_json, f)


# Compute Fisher information for a client
def compute_fisher(model, dataloader, criterion):
    fisher = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    model.eval()

    for data, labels in dataloader:
        data, labels = data.to('cpu'), labels.to('cpu')
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2)

    # Average Fisher information across batches
    fisher = {name: value / len(dataloader) for name, value in fisher.items()}
    return fisher