import multiprocessing
import torch
import time

class Client(multiprocessing.Process):
    def __init__(self, client_id, device, dataset):
        super(Client, self).__init__()
        self.client_id = client_id
        self.device = device
        self.dataset = dataset

    def run(self):
        print(f"Client {self.client_id} started on device {self.device} with dataset {self.dataset}.")
        # Simulate some work with sleep
        time.sleep(2)
        print(f"Client {self.client_id} finished working.")

# Assume clientsDict is defined elsewhere
clientsDict = {0: 'Dataset1', 1: 'Dataset2', 2: 'Dataset3'}

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # Use "spawn" start method

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_clients = 3
    clients = [Client(client_id=i, device=device, dataset=clientsDict[i]) for i in range(num_clients)]

    # Start all clients
    for client in clients:
        client.start()

    # Wait for all clients to finish
    for client in clients:
        client.join()

    print("All clients have finished.")
