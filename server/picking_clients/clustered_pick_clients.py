import random
import numpy as np


def clustered_pick_clients(initial_data, rng):
    '''
    adjusted_round                             |  0 1 2 3 4 5 6
    divide by (len of clustered_clients_list)  |  2 2 2 2 2 2 2
    remainder (cluster_to_pick)                |  0 1 0 1 0 1 0
    '''

    adjusted_round = int(initial_data['curRound']) + int(initial_data['initial_cluster'])
    cluster_to_pick = adjusted_round % len(initial_data['clustered_clients_list'])
    picked_cluster = initial_data['clustered_clients_list'][cluster_to_pick]

    pickedClients = rng.choice(picked_cluster, initial_data['updateClientsPerRound'], replace=False)
    print(f'cluster picked from -> {cluster_to_pick}')

    return pickedClients