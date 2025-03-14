import os
from typing import Any, Dict, List
import torch
import copy
from server.fedOptimizer.fedOptParent import fedOptParent


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not weights:
        raise ValueError("The weights list is empty.")

    new_state_dict = {}
    for key in weights[0].keys():
        stacked = torch.stack([client[key] for client in weights], dim=0)
        new_state_dict[key] = torch.mean(stacked.float(), dim=0)

    return new_state_dict

class fed_2way_avg(fedOptParent):
    def __init__(self, rootModel, cudaId, additionalInfo=None):
        super().__init__(rootModel, cudaId, additionalInfo)
        self.cluster_info = additionalInfo.get('cluster_info', None) if additionalInfo else None
        self.picked_clients = additionalInfo.get('picked_clients', None) if additionalInfo else None
        self.num_cluster = additionalInfo.get('num_cluster', None) if additionalInfo else None
        
    def aggregate_by_cluster(self):
        """Aggregate models within each cluster separately"""
        if not all([self.cluster_info, self.picked_clients, self.num_cluster]):
            return
            
        # Group client models by cluster
        cluster_models = {}
        for client_idx, model in enumerate(self.clientsModels):
            client = self.picked_clients[client_idx]
            cluster = self.num_cluster[client_idx]
            
            if cluster not in cluster_models:
                cluster_models[cluster] = []
            cluster_models[cluster].append(model)
            
        # Aggregate models within each cluster
        sub_root_path = f'{self.resultPath}/sub_roots'
        os.makedirs(sub_root_path, exist_ok=True)
        
        for cluster, models in cluster_models.items():
            # Average weights for this cluster
            cluster_weights = average_weights(models)
            
            # Create a new model instance for this cluster
            sub_root = copy.deepcopy(self.rootModel)
            sub_root.load_state_dict(cluster_weights)
            
            # Save sub-root model for this cluster

            rootModelPath = self.basicConfig['rootModelFilePath']
            testName = self.basicConfig['testName']
            torch.save(
                sub_root.state_dict(), 
                f'{rootModelPath}/sub_{self.clientType}_rootModel-{testName}.pth'
            )
            
    def aggregate(self):
        # First, perform cluster-wise aggregation
        self.aggregate_by_cluster()
        
        # Then perform global aggregation as before
        updated_weights = average_weights(self.clientsModels)
        self.resultRootModel.load_state_dict(updated_weights)

        return self.resultRootModel