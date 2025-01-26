from typing import Any, Dict, List
import torch
import copy
from server.fedOptimizer.fedOptParent import fedOptParent


def weighted_fed_average(
    client_weights: List[Dict[str, torch.Tensor]],
    root_weights: Dict[str, torch.Tensor],
    device,
    eps: float = 1e-8
    ) -> Dict[str, torch.Tensor]:

    if not client_weights:
        raise ValueError("client_weights is empty.")

    new_state_dict = {}

    # --------------------------
    # 1) 각 클라이언트 내부에서 "전체 파라미터 변화량 합" 계산
    #    { client_idx : total_diff_value }
    # --------------------------
    clients_total_diff = []
    for cw in client_weights:
        total_diff = 0.0
        # 모든 파라미터에 대한 L1 변화량 합
        for param_key in cw.keys():
            diff = cw[param_key] - root_weights[param_key]
            total_diff += diff.abs().sum().item()
        clients_total_diff.append(total_diff + eps)  # eps 더하기

    # --------------------------
    # 2) 파라미터별로 반복하면서,
    #    "각 클라이언트에서의 ratio"로 가중합
    # --------------------------
    # client_weights[0]이 기준이라고 가정(모든 키 동일)
    param_keys = list(client_weights[0].keys())

    for param_key in param_keys:
        # 클라이언트별 ratio와 파라미터 값들을 모아 최종 가중합을 계산
        ratio_list = []
        weighted_sum = None

        # 먼저 모든 클라이언트 c에 대해 ratio_{c, p} 계산
        for c_idx, cw in enumerate(client_weights):
            diff = cw[param_key] - root_weights[param_key]
            mag_value = diff.abs().sum().item()  # L1 norm
            ratio_c_p = mag_value / clients_total_diff[c_idx]  # 비중

            ratio_list.append(ratio_c_p)

        # ratio를 이용해 가중합
        for c_idx, cw in enumerate(client_weights):
            # 텐서에 ratio 곱
            part = cw[param_key] * ratio_list[c_idx]
            if weighted_sum is None:
                weighted_sum = part
            else:
                weighted_sum += part

        # ratio 합
        sum_ratios = sum(ratio_list)
        if sum_ratios < eps:
            # 모든 클라이언트에서 diff=0인 극단적 경우엔 root_weights 사용
            new_state_dict[param_key] = root_weights[param_key].clone()
        else:
            # 최종 "가중 평균" = weighted_sum / sum_ratios
            new_state_dict[param_key] = weighted_sum / sum_ratios

    return new_state_dict


class weighed_fed_avg_param_diff(fedOptParent):
    def __init__(self, rootModel, cudaId, additionalInfo=None):
        super().__init__(rootModel, cudaId, additionalInfo)

    def aggregate(self):

        # 1) 클라이언트 모델 파라미터를 device로 이동
        for i, cw in enumerate(self.clientsModels):
            for param_key, param_value in cw.items():
                cw[param_key] = param_value.to(self.device)

        # 2) 루트 모델 자체를 device로 이동
        self.resultRootModel.to(self.device)

        # Update server model based on clients models
        updated_weights = weighted_fed_average(
            client_weights=self.clientsModels,
            root_weights=self.resultRootModel.state_dict(),
            device=self.device
        )
        self.resultRootModel.load_state_dict(updated_weights)

        # Update average loss of this round
        '''
        avg_loss = sum(clients_losses) / len(clients_losses)
        train_losses.append(avg_loss)
        '''

        return self.resultRootModel
