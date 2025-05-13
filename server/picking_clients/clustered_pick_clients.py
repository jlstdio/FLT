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


def multi_cluster_pick_clients(initial_data, rng, num_clusters_to_pick, num_clients_per_cluster, cluster_class_distributions=None):
    """
    Pick clients from multiple clusters where clusters are formed based on client ID's ones digit.
    
    Args:
        initial_data: Dictionary containing necessary data
        rng: Random number generator
        num_clusters_to_pick (int): Number of clusters to pick (M)
        num_clients_per_cluster (int): Number of clients to pick per cluster (N/M)
        cluster_class_distributions: Not used in this implementation
    
    Returns:
        list: List of picked client IDs
        int: Number of the selected cluster
    """
    # Create clusters based on ones digit of client IDs (0-9)
    digit_based_clusters = [[] for _ in range(10)]  # 10 clusters for digits 0-9

    type_info_by_clients = {}
    
    # Populate clusters based on ones digit
    clients = []
    for cluster in initial_data["clustered_clients_list"]:
        clients.extend(cluster)
    
    for client_id in clients:
        ones_digit = client_id % 10
        digit_based_clusters[ones_digit].append(client_id)
    
    # Filter out empty clusters
    digit_based_clusters = [cluster for cluster in digit_based_clusters if cluster]
    
    # Randomly select M clusters
    total_clusters = len(digit_based_clusters)
    if num_clusters_to_pick > total_clusters:
        num_clusters_to_pick = total_clusters
    
    selected_cluster_indices = rng.choice(total_clusters, num_clusters_to_pick, replace=False)
    selected_clusters = [digit_based_clusters[i] for i in selected_cluster_indices]
    
    # Pick N/M clients from each selected cluster
    picked_clients = []
    for cluster, idx in zip(selected_clusters, selected_cluster_indices):
        # If the cluster has fewer clients than needed, take all of them
        count = min(num_clients_per_cluster, len(cluster))
        chosen_clients = rng.choice(cluster, count, replace=False).tolist()
        picked_clients.extend(chosen_clients)
        for client_id in chosen_clients:
            print(f'client {client_id} is picked from cluster {idx}')
            type_info_by_clients[client_id] = idx

    
    return picked_clients, selected_clusters, type_info_by_clients   # -1 for cluster number as we're using multiple clusters
