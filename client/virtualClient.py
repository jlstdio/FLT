from multiprocessing import Process
from torch import optim, nn
from model.testModel import SimpleNN
import torch
import os


class Client(Process):
    def __init__(self, client_id, device):
        super(Client, self).__init__()
        self.client_id = client_id
        self.device = device
        self.model = SimpleNN().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()

    def train(self, data, targets, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            print(f"Client {self.client_id} Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    def run(self):
        print(f"Client {self.client_id} with PID {os.getpid()} started.")
        # Example: Perform some operations with the model
        dummy_input = torch.randn(10, 10).to(self.device)
        dummy_target = torch.randn(10, 1).to(self.device)
        self.train(dummy_input, dummy_target)
        print(f"Client {self.client_id} finished training.")
