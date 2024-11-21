import os
import random
import shutil
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
            maximum_pth_to_mix = self.additionalInfo['maximum_pth_to_mix']
            selected_files = random.sample(eligible_files, min(maximum_pth_to_mix, len(eligible_files)))

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

    def afterWork(self):
        M = self.additionalInfo['server_round_mem']
        pth_files = self.additionalInfo['pth_files']
        memorized_pth_path = self.additionalInfo['memorized_pth_path']
        curRound = self.additionalInfo['curRound']

        # Aggregate 끝난 후 -> 이번 round 에 가져온 models들 self.basicConfig['memorizedPthPath]의 위치에 옮기기
        # 이번 round에 가져온 모델들을 memorizedPthPath로 이동
        for filePath in pth_files:
            fileName = os.path.basename(filePath)
            client_id = int(fileName.split('_')[0])
            new_file_name = f"{client_id}_round{curRound}.pth"
            destination = os.path.join(memorized_pth_path, new_file_name)
            shutil.move(filePath, destination)
            print(f"Moved {filePath} to {destination}")

        # memorizedPthPath에서 오래된 모델 삭제 (최신 M 라운드만 유지)
        all_memorized_files = [
            os.path.join(memorized_pth_path, f) for f in os.listdir(memorized_pth_path) if f.endswith('.pth')
        ]

        # 파일별 라운드 번호 추출
        files_with_round = []
        for mem_file in all_memorized_files:
            mem_file_name = os.path.basename(mem_file)
            try:
                parts = mem_file_name.split('_round')
                round_num = int(parts[1].replace('.pth', ''))
                files_with_round.append((mem_file, round_num))
            except (IndexError, ValueError):
                print(f"Invalid memorized file name format: {mem_file_name}")
                continue

        # 라운드 번호 기준으로 정렬 (오래된 순)
        files_with_round.sort(key=lambda x: x[1])

        # 유지할 라운드 번호 범위
        min_round_to_keep = curRound - M + 1

        # 삭제할 파일 찾기
        files_to_delete = [f for f, r in files_with_round if r < min_round_to_keep]

        for filePath in files_to_delete:
            os.remove(filePath)
            print(f"Deleted old memorized model: {filePath}")


'''
unpacker = fedAvg()
unpacker.loadPth('../pth/gesture_transformer_epoch233.pth')
unpacker.summary()
'''