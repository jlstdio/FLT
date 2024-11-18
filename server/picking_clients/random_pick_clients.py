import numpy as np


def random_pick_clients(initial_data, seed):
    np.random.seed(seed)

    clients_per_round = initial_data['clients_per_round']
    total_clients = initial_data['total_clients']

    pickedClients = np.random.choice(total_clients, clients_per_round, replace=False)

    return pickedClients
