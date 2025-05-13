import random
import numpy as np


def mixed_clustered_pick_clients(initial_data, rng, random_cluster_pick=False, rounds_per_cluster_change=1, num_clusters_to_pick=2):
    
    total_clusters = len(initial_data['clustered_clients_list'])
    clients_per_cluster = initial_data['updateClientsPerRound'] // num_clusters_to_pick
    picked_clusters = []
    all_picked_clients = []
    type_info_by_clients = {}
    
    if random_cluster_pick:
        # Randomly pick multiple clusters without replacement
        picked_cluster_indices = rng.choice(
            total_clusters, 
            num_clusters_to_pick, 
            replace=False
        )
        
    else:
        # Calculate base cluster index from current round
        base_cluster = (int(initial_data['curRound']) // rounds_per_cluster_change + int(initial_data['initial_cluster'])) % total_clusters
        
        # Generate sequence of cluster indices
        picked_cluster_indices = [(base_cluster + i) % total_clusters 
                                for i in range(num_clusters_to_pick)]
    
    # Pick clients from each selected cluster
    for cluster_idx in picked_cluster_indices:
        cluster = initial_data['clustered_clients_list'][cluster_idx]
        picked_clients = rng.choice(cluster, clients_per_cluster, replace=False)
        all_picked_clients.extend(picked_clients)
        picked_clusters.append(cluster_idx)

        for idx in picked_clients:
            type_info_by_clients[idx] = cluster_idx
    
    print(f'clusters picked from -> {picked_clusters}')
    
    return np.array(all_picked_clients), picked_clusters, type_info_by_clients


def mixed_clustered_pick_clients_with_initial_cluster(
    initial_data, rng, 
    initial_n_rounds, initial_cluster_idx,
    random_cluster_pick=False, rounds_per_cluster_change=1, num_clusters_to_pick=2
):
    """
    첫 initial_n_rounds까지는 initial_cluster_idx에서만 클라이언트를 뽑고,
    이후에는 mixed_clustered_pick_clients와 동일하게 동작합니다.
    """

    cur_round = int(initial_data['curRound'])
    if cur_round <= initial_n_rounds:
        cluster = initial_data['clustered_clients_list'][initial_cluster_idx]
        clients_per_cluster = initial_data['updateClientsPerRound']
        picked_clients = rng.choice(cluster, clients_per_cluster, replace=False)
        picked_clusters = [initial_cluster_idx]
        type_info_by_clients = {idx: initial_cluster_idx for idx in picked_clients}
        print(f'clusters picked from -> {picked_clusters} (initial phase)')
        return np.array(picked_clients), picked_clusters, type_info_by_clients
    else:
        total_clusters = len(initial_data['clustered_clients_list'])
        clients_per_cluster = initial_data['updateClientsPerRound'] // num_clusters_to_pick
        picked_clusters = []
        all_picked_clients = []
        type_info_by_clients = {}

        if random_cluster_pick:
            # Randomly pick multiple clusters without replacement
            picked_cluster_indices = rng.choice(
                total_clusters, 
                num_clusters_to_pick, 
                replace=False
            )
            
        else:
            # Calculate base cluster index from current round
            base_cluster = (int(initial_data['curRound']) // rounds_per_cluster_change + int(initial_data['initial_cluster'])) % total_clusters
            
            # Generate sequence of cluster indices
            picked_cluster_indices = [(base_cluster + i) % total_clusters 
                                    for i in range(num_clusters_to_pick)]


        # Pick clients from each selected cluster
        for cluster_idx in picked_cluster_indices:
            cluster = initial_data['clustered_clients_list'][cluster_idx]
            picked_clients = rng.choice(cluster, clients_per_cluster, replace=False)
            all_picked_clients.extend(picked_clients)
            picked_clusters.append(cluster_idx)

            for idx in picked_clients:
                type_info_by_clients[idx] = cluster_idx
        
        print(f'clusters picked from -> {picked_clusters}')
        
        return np.array(all_picked_clients), picked_clusters, type_info_by_clients
