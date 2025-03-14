import random
import numpy as np


def mixed_clustered_pick_clients(initial_data, rng, random_cluster_pick=False, rounds_per_cluster_change=1, num_clusters_to_pick=2):
    
    total_clusters = len(initial_data['clustered_clients_list'])
    clients_per_cluster = initial_data['updateClientsPerRound'] // num_clusters_to_pick
    picked_clusters = []
    all_picked_clients = []
    
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
    
    print(f'clusters picked from -> {picked_clusters}')
    
    return np.array(all_picked_clients), picked_clusters