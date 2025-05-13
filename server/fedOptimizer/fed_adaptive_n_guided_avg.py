import os
from typing import Any, Dict, List
from model.testModel_wo_softmax_3_layer import testNN_wo_Softmax_3_layer
import torch
import copy
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset
from server.fedOptimizer.fedOptParent import fedOptParent


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not weights:
        raise ValueError("The weights list is empty.")

    new_state_dict = {}
    for key in weights[0].keys():
        stacked = torch.stack([client[key] for client in weights], dim=0)
        new_state_dict[key] = torch.mean(stacked.float(), dim=0)

    return new_state_dict

class fed_adaptive_avg(fedOptParent):
    def __init__(self, rootModel, cudaId, additionalInfo=None):
        super().__init__(rootModel, cudaId, additionalInfo)

        if additionalInfo and 'rootModelFilePath' in additionalInfo:
            self.latest_models_path = additionalInfo['latest_sub_roots_path']
            os.makedirs(self.latest_models_path, exist_ok=True)
        
        self.guided_ds = additionalInfo['guideDataset_list']
        
    def mix_models(self, global_state, cluster_state, mix_ratio):
        """
        Mix global and cluster models with given ratio
        mix_ratio: portion of global model in the final mix (0~1)
        """
        mixed_state = {}
        for key in global_state.keys():
            mixed_state[key] = mix_ratio * global_state[key] + (1 - mix_ratio) * cluster_state[key]
        return mixed_state
    
    def update_latest_sub_roots(self, cluster_type, model_state):
        """Update the latest sub-root model for given cluster type"""
        if not hasattr(self, 'latest_models_path'):
            return
            
        testName = self.additionalInfo['testName']
        latest_path = os.path.join(
            self.latest_models_path, 
            f'sub_{cluster_type}_rootModel-{testName}.pth'
        )
        torch.save(model_state, latest_path)

    def get_all_latest_sub_roots(self, current_clusters):
        """Get all latest sub-root models except current round's clusters"""
        if not hasattr(self, 'latest_models_path') or not os.path.exists(self.latest_models_path):
            return []

        latest_models = []
        used_clusters = []  # Track which clusters are being used
        testName = self.additionalInfo['testName']
        
        # Check if any valid model files exist
        model_files = [f for f in os.listdir(self.latest_models_path) 
                    if f.endswith(f'-{testName}.pth')]
        
        if not model_files:  # If no files exist, return empty list
            return []
            
        for file in model_files:
            cluster_type = int(file.split('_')[1].split('rootModel')[0])
            if cluster_type not in current_clusters:  # Only load if not in current round
                model_path = os.path.join(self.latest_models_path, file)
                model_state = torch.load(model_path)
                latest_models.append(model_state)
                used_clusters.append(cluster_type)

        if used_clusters:
            print(f"[Additional Aggregation] Using previous sub-root models from clusters: {sorted(used_clusters)}")
        
        return latest_models

    def aggregate(self):
        # First aggregation with current round's models
        global_weights = average_weights(self.clientsModels)
        self.resultRootModel.load_state_dict(global_weights)
        
        # Get current round's cluster types and create sub-root models
        type_info_by_clients = self.additionalInfo['cluster_info']
        current_clusters = set()
        cluster_models = {}
        
        for model, client_id in zip(self.clientsModels, self.clients_ids):
            cluster_type = type_info_by_clients[client_id]
            current_clusters.add(cluster_type)
            if cluster_type not in cluster_models:
                cluster_models[cluster_type] = []
            cluster_models[cluster_type].append(model)
            
        # Create and save sub-root models for current round
        mix_ratio = self.additionalInfo.get('global_mix_ratio', 0.0)
        rootModel = copy.deepcopy(self.rootModelStatic)
        
        for cluster, models in cluster_models.items():
            print(f'[TEST] aggregating sub-root models for cluster {cluster} with {len(models)} clients')
            cluster_weights = average_weights(models)
            
            if mix_ratio > 0.0:
                mixed_weights = self.mix_models(global_weights, cluster_weights, mix_ratio)
                sub_root = copy.deepcopy(rootModel)
                sub_root.load_state_dict(mixed_weights)
            else:
                sub_root = copy.deepcopy(rootModel)
                sub_root.load_state_dict(cluster_weights)
            
            # Save to both regular and latest locations
            rootModelPath = self.additionalInfo['rootModelFilePath']
            testName = self.additionalInfo['testName']
            
            # Save regular sub-root
            torch.save(
                sub_root.state_dict(), 
                f'{rootModelPath}/sub_{cluster}_rootModel-{testName}.pth'
            )

            saved_pth_path = self.additionalInfo['resultPath'] + '/score/server/sub_root_models'
            os.makedirs(saved_pth_path, exist_ok=True)
            torch.save(sub_root.state_dict(), f'{saved_pth_path}/sub_{cluster}_rootModel.pth')
            '''
            IMPLEMENTATION JUST FOR EXPERIEMENTAL PURPOSES
            '''
            round = self.additionalInfo['curRound']
            os.makedirs(f'{saved_pth_path}/history', exist_ok=True)
            torch.save(sub_root.state_dict(), f'{saved_pth_path}/history/sub_{cluster}_round{round}_rootModel.pth')
            
            # Update latest sub-root
            self.update_latest_sub_roots(cluster, sub_root.state_dict())

        ###########################
        # guided learning process #
        ###########################
        # select anchor model
        anchor_model_idx = 3 # cifar-10_r1
        classNum = 10
        train_up_to_idx = 0

        # load the anchor model
        anchor_model = copy.deepcopy(self.resultRootModel)

        # make new model and load anchor model
        adapter_model_pah = f'{rootModelPath}/anchor_{anchor_model_idx}_rootModel-{testName}.pth'        
        if os.path.exists(adapter_model_pah):
            adapter_model = testNN_wo_Softmax_3_layer(classNum).to(self.device)
            adapter_model.load_state_dict(torch.load(adapter_model_pah))

        # replace anchor model conv layer with adapter model conv layer
        # it could be varied based on the adapterEndPoint
        # for example, if adapterEndPoint = 'conv2', then replace conv2
        anchor_model.conv1 = copy.deepcopy(adapter_model.conv1)

        for idx, (name, param) in enumerate(anchor_model.named_parameters()):
                    if idx <= train_up_to_idx:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        # use only one dataset list
        single_guide_ds = self.guided_ds[anchor_model_idx] # cifar-10_r1

        y_list, x_list = zip(*single_guide_ds)
        y_list, x_list = np.array(y_list), np.array(x_list)

        x_list = torch.tensor(x_list, dtype=torch.float32).permute(0, 3, 1, 2)
        y_list = torch.tensor(y_list, dtype=torch.long)
        
        guided_dataset = TensorDataset(x_list, y_list)
        self.guide_ds_loader = DataLoader(guided_dataset, 
                                       batch_size=64, 
                                       shuffle=True)
        
        self.optimizer = optim.SGD(self.resultRootModel.parameters(), lr=1e-3)

        self.criterion = nn.CrossEntropyLoss()
        
        anchor_model.train()
        
        for epoch in range(2):
            running_loss = 0.0
            for inputs, labels in self.guide_ds_loader:
                inputs, labels = inputs.to(self.cudaId), labels.to(self.cudaId)
                anchor_model.zero_grad()
                outputs = anchor_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            # print(f"Epoch {epoch+1}, Loss: {running_loss/len(self.guide_ds_loader)}")

        for idx, (name, param) in enumerate(anchor_model.named_parameters()):
                    if idx <= train_up_to_idx:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        for epoch in range(5):
            running_loss = 0.0
            for inputs, labels in self.guide_ds_loader:
                inputs, labels = inputs.to(self.cudaId), labels.to(self.cudaId)
                anchor_model.zero_grad()
                outputs = anchor_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            # print(f"Epoch {epoch+1}, Loss: {running_loss/len(self.guide_ds_loader)}")

        return anchor_model