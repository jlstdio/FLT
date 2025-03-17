def pick_clients(self):
        pickedClients = []
        numCluster = 0
        type_info_by_clients = None ## 

        if self.serverConfig['pickMode'] == 'random':
            from server.picking_clients.random_pick_clients import random_pick_clients

            initial_data = {
                "clients_per_round": self.updateClientsPerRound,
                "total_clients": self.basicConfig['numClient']
            }
            pickedClients = random_pick_clients(initial_data, self.rng)
        elif self.serverConfig['pickMode'] == 'sequential':
            from server.picking_clients.sequential_pick_clients import sequential_pick_clients

            initial_data = {
                "pair_size": self.updateClientsPerRound,
                "total_clients": self.basicConfig['numClient'],
                "curRound": self.currentRound.value,
                "initial_idx": 3
            }
            pickedClients = sequential_pick_clients(initial_data)
        elif self.serverConfig['pickMode'] == 'pickey':
            from server.picking_clients.pickey_pick_clients import pickey_pick_clients

            initial_data = {"none": None}
            pickedClients = pickey_pick_clients(initial_data, self.rng)
        elif self.serverConfig['pickMode'] == 'clustered_sequential' or self.serverConfig['pickMode'] == 'clustered':
            from server.picking_clients.clustered_pick_clients import clustered_pick_clients

            participantInfo = self.basicConfig['participantsInfo']
            past_idx = 0
            cluster_list = []
            for typeInfo in participantInfo:
                type_id = typeInfo.split(':')[0]
                type_ratio = float(typeInfo.split(':')[1])
                next_idx = past_idx + int(len(self.clientsList) * type_ratio)
                cluster_list.append(self.clientsList[past_idx:next_idx])
                past_idx = next_idx

            initial_data = {
                "clustered_clients_list": cluster_list,
                "updateClientsPerRound": self.basicConfig['updateClientsPerRound'],
                "curRound": self.currentRound.value,
                "initial_cluster": 0
            }
            pickedClients, numCluster = clustered_pick_clients(initial_data, self.rng, self.serverConfig['update_cluster_every'])

        elif self.serverConfig['pickMode'] == 'clustered_random':
            from server.picking_clients.clustered_pick_clients import clustered_pick_clients

            participantInfo = self.basicConfig['participantsInfo']
            past_idx = 0
            cluster_list = []
            for typeInfo in participantInfo:
                type_id = typeInfo.split(':')[0]
                type_ratio = float(typeInfo.split(':')[1])
                next_idx = past_idx + int(len(self.clientsList) * type_ratio)
                cluster_list.append(self.clientsList[past_idx:next_idx])
                past_idx = next_idx

            initial_data = {
                "clustered_clients_list": cluster_list,
                "updateClientsPerRound": self.basicConfig['updateClientsPerRound'],
                "curRound": self.currentRound.value,
                "initial_cluster": 0
            }
            pickedClients, numCluster = clustered_pick_clients(initial_data, self.rng, self.serverConfig['update_cluster_every'])

        elif self.serverConfig['pickMode'] == 'mixed_clustered':
            from server.picking_clients.mixed_clustered_pick_clients import mixed_clustered_pick_clients
    
            # Create cluster list based on participant info
            participantInfo = self.basicConfig['participantsInfo']
            past_idx = 0
            cluster_list = []
            for typeInfo in participantInfo:
                type_id = typeInfo.split(':')[0]
                type_ratio = float(typeInfo.split(':')[1])
                next_idx = past_idx + int(len(self.clientsList) * type_ratio)
                cluster_list.append(self.clientsList[past_idx:next_idx])
                past_idx = next_idx

            initial_data = {
                "clustered_clients_list": cluster_list,
                "updateClientsPerRound": self.basicConfig['updateClientsPerRound'],
                "curRound": self.currentRound.value,
                "initial_cluster": 0
            }   
    
            num_clusters_to_pick = self.serverConfig.get('num_clusters_to_pick', 2)
            
            pickedClients, numCluster, type_info_by_clients = mixed_clustered_pick_clients(
                initial_data=initial_data,
                rng=self.rng,
                random_cluster_pick=False,
                rounds_per_cluster_change=self.serverConfig['update_cluster_every'],
                num_clusters_to_pick=num_clusters_to_pick)

        # self.update_picked_clients(pickedClients, numCluster)
        return pickedClients, numCluster, type_info_by_clients