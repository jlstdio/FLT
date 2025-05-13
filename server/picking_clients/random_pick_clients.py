import numpy as np


def random_pick_clients(initial_data, rng):

    clients_per_round = initial_data['clients_per_round'] # type: int
    total_clients = initial_data['total_clients']

    # print(f'total: {total_clients} | clients_per_round: {clients_per_round}')
    pickedClients = rng.choice(total_clients, clients_per_round, replace=False)

    return pickedClients


def init_pick_and_random_pick_clients(initial_data, rng, initial_n_rounds, initial_cluster_idx):

    clients_per_round = initial_data['clients_per_round'] # type: int
    total_clients = initial_data['total_clients']

    cur_round = int(initial_data['curRound'])
    if cur_round <= initial_n_rounds:
        cluster = initial_data['clustered_clients_list'][initial_cluster_idx]
        clients_per_cluster = initial_data['clients_per_round']
        pickedClients = rng.choice(cluster, clients_per_cluster, replace=False)
        print(f'clusters picked from -> {pickedClients} (initial phase)')
    else:
        # print(f'total: {total_clients} | clients_per_round: {clients_per_round}')
        pickedClients = rng.choice(total_clients, clients_per_round, replace=False)

    return pickedClients
