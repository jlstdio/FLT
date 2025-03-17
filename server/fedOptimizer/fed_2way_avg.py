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
        
    def aggregate_by_cluster(self):
        
        type_info_by_clients =  self.additionalInfo['cluster_info']
        rootModel = copy.deepcopy(self.rootModelStatic)
            
        # Group client models by cluster
        cluster_models = {}
        for model, client_id in zip(self.clientsModels, self.clients_ids):
            cluster_type = type_info_by_clients[client_id]

            if cluster_type not in cluster_models:
                cluster_models[cluster_type] = []
            cluster_models[cluster_type].append(model)
            print(f'[TEST] client_id: {client_id}, cluster_type: {cluster_type}')
        
        for cluster, models in cluster_models.items():
            # Average weights for this cluster
            cluster_weights = average_weights(models)
            
            # Create a new model instance for this cluster
            sub_root = copy.deepcopy(rootModel)
            sub_root.load_state_dict(cluster_weights)
            
            # Save sub-root model for this cluster
            rootModelPath = self.additionalInfo['rootModelFilePath']
            testName = self.additionalInfo['testName']
            torch.save(
                sub_root.state_dict(), 
                f'{rootModelPath}/sub_{cluster}_rootModel-{testName}.pth'
            )
            
    def aggregate(self):

        self.cluster_info = self.additionalInfo.get('cluster_info', None) if self.additionalInfo else None
        self.picked_clients = self.additionalInfo.get('picked_clients', None) if self.additionalInfo else None        
        
        # First, perform cluster-wise aggregation
        self.aggregate_by_cluster()
        
        # Then perform global aggregation as before
        updated_weights = average_weights(self.clientsModels)
        self.resultRootModel.load_state_dict(updated_weights)

        return self.resultRootModel