import numpy as np


def random_pick_clients(initial_data, rng):

    clients_per_round = initial_data['clients_per_round']
    total_clients = initial_data['total_clients']

    # print(f'total: {total_clients} | clients_per_round: {clients_per_round}')
    pickedClients = rng.choice(total_clients, clients_per_round, replace=False)

    return pickedClients
