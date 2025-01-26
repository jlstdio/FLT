import random
import numpy as np


def clustered_pick_clients(initial_data, rng, random_cluster_pick=False, rounds_per_cluster_change=1):
    '''
    adjusted_round                             |  0 1 2 3 4 5 6
    divide by (len of clustered_clients_list)  |  2 2 2 2 2 2 2
    remainder (cluster_to_pick)                |  0 1 0 1 0 1 0
    rounds_per_cluster_change                  |  N  N  N  N  N
    '''

    if random_cluster_pick:
        num_of_clusters = len(initial_data['clustered_clients_list'])
        cluster_to_pick = rng.choice(num_of_clusters, 1, replace=False)[0]
        picked_cluster = initial_data['clustered_clients_list'][cluster_to_pick]
        pickedClients = rng.choice(picked_cluster, initial_data['updateClientsPerRound'], replace=False)

    else:
        # 현재 라운드를 N으로 나누어 N번에 한 번 클러스터 변경
        adjusted_round = (int(initial_data['curRound']) // rounds_per_cluster_change) + int(initial_data['initial_cluster'])
        cluster_to_pick = adjusted_round % len(initial_data['clustered_clients_list'])
        picked_cluster = initial_data['clustered_clients_list'][cluster_to_pick]

        pickedClients = rng.choice(picked_cluster, initial_data['updateClientsPerRound'], replace=False)

    print(f'cluster picked from -> {cluster_to_pick}')

    return pickedClients, cluster_to_pick
