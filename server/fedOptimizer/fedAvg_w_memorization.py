import os
import random
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
        new_state_dict[key] = torch.mean(stacked, dim=0)

    return new_state_dict


class fedAvg_w_mem(fedOptParent):
    def __init__(self, rootModel, cudaId, additionalInfo):
        super().__init__(rootModel, cudaId, additionalInfo)

    def aggregate(self):
        memorized_pth_path = self.additionalInfo['memorized_pth_path']
        if len(os.listdir(memorized_pth_path)) > 0:
            os.makedirs(memorized_pth_path, exist_ok=True)

            # memorizedPthPath에서 모든 pth 파일 가져오기
            memorized_files = [os.path.join(memorized_pth_path, f) for f in os.listdir(memorized_pth_path) if
                               f.endswith('.pth')]
            print(f'memorized_files')
            print(memorized_files)

            eligible_files = []

            for mem_file in memorized_files:
                mem_file_name = os.path.basename(mem_file)
                try:
                    parts = mem_file_name.split('_round')
                    client_session_id = int(parts[0])

                    eligible_files.append(mem_file)

                except (IndexError, ValueError):
                    print(f"Invalid memorized file name format: {mem_file_name}")
                    continue

            print(f'eligible_files')
            print(eligible_files)

            # 최대 'maximum_pth_to_mix'개 랜덤 선택
            max_mix = self.serverConfig.get('maximum_pth_to_mix', self.serverConfig['maximum_pth_to_mix'])  # 기본값 10 설정
            selected_files = random.sample(eligible_files, min(max_mix, len(eligible_files)))

            print(f'selected_files')
            print(selected_files)

            for filePath in selected_files:
                self.registerPth(filePath)
                print(f"Registered memorized model: {filePath}")

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