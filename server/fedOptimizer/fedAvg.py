from typing import Any, Dict, List
import torch
import copy
from server.fedOptimizer.fedOptParent import fedOptParent


def average_weights(weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights_avg = copy.deepcopy(weights[0])

    for key in weights_avg.keys():
        for i in range(1, len(weights)):
            weights_avg[key] += weights[i][key]
        weights_avg[key] = torch.div(weights_avg[key], len(weights))

    return weights_avg


class fedAvg(fedOptParent):
    def __init__(self, rootModel, cudaId):
        super().__init__(rootModel, cudaId)

    def aggregate(self):
        # Update server model based on clients models
        updated_weights = average_weights(self.clientsModels)
        self.resultRootModel.load_state_dict(updated_weights)

        # Update average loss of this round
        '''
        avg_loss = sum(clients_losses) / len(clients_losses)
        train_losses.append(avg_loss)
        '''

        return self.resultRootModel

'''
unpacker = fedAvg()
unpacker.loadPth('../pth/gesture_transformer_epoch233.pth')
unpacker.summary()
'''