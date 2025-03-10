def pick_clients(self):
        pickedClients = []
        numCluster = 0

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

        # self.update_picked_clients(pickedClients, numCluster)
        return pickedClients, numCluster