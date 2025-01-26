import torch
from typing import List, Dict
from server.fedOptimizer.fedOptParent import fedOptParent


def weighted_average_with_fisher(
    client_weights: List[Dict[str, torch.Tensor]],
    client_fishers: List[Dict[str, torch.Tensor]],
    eps: float = 1e-8
) -> Dict[str, torch.Tensor]:

    if not client_weights:
        raise ValueError("client_weights is empty.")
    if len(client_weights) != len(client_fishers):
        raise ValueError("The number of client_weights and client_fishers must match.")

    # 첫 번째 클라이언트의 파라미터 키를 기준으로 삼음 (모든 클라이언트의 키가 동일하다고 가정)
    param_keys = list(client_weights[0].keys())
    new_state_dict = {}

    for param_key in param_keys:
        # 같은 shape의 텐서들끼리 element-wise로 합산할 것이므로 None으로 초기화
        weighted_sum = None
        fisher_sum = None

        # 모든 클라이언트 c에 대해
        for c_idx, cw in enumerate(client_weights):
            # 현재 클라이언트의 파라미터 값과 Fisher 정보
            param_value = cw[param_key]              # 모델 파라미터 (Tensor)
            fisher_value = client_fishers[c_idx][param_key]  # Fisher 정보 (Tensor, param_value와 동일 shape 가정)

            # 가중치가 fisher_value (즉, 중요도가 높을수록 반영 비중↑)
            contribution = param_value * fisher_value

            if weighted_sum is None:
                weighted_sum = contribution.clone()
                fisher_sum = fisher_value.clone()
            else:
                weighted_sum += contribution
                fisher_sum += fisher_value

        # 모든 클라이언트의 기여(contribution)를 모은 뒤, fisher_sum으로 나눠 최종 결정
        new_state_dict[param_key] = weighted_sum / (fisher_sum + eps)

    return new_state_dict


class weighed_fed_avg_fisher(fedOptParent):
    """
    Fisher 정보를 통해 파라미터 중요도를 반영해 FedAvg 비슷한 과정을 구현한 예시 클래스.
    """
    def __init__(self, rootModel, cudaId, additionalInfo=None):
        super().__init__(rootModel, cudaId, additionalInfo)
        # 클라이언트로부터 수집한 Fisher 정보를 저장할 리스트
        self.clientsFisher = []

    def aggregate(self):
        """
        1) 클라이언트들이 학습한 파라미터(self.clientsModels)와 Fisher(self.clientsFisher)를 이용해
           중요도 기반 가중평균을 수행.
        2) 그 결과를 서버(root) 모델에 반영해 업데이트.
        3) Fisher 자체는 별도로 평균 낼 수도 있고, 필요에 따라 추가적인 처리를 진행.
        """
        # (1) Fisher-weighted average
        updated_weights = weighted_average_with_fisher(self.clientsModels, self.clientsFisher)
        self.resultRootModel.load_state_dict(updated_weights)

        # (2) Fisher도 필요하다면 평균 혹은 다른 방식으로 업데이트
        #     아래 예시에선 평균 처리를 했지만, 필요에 맞게 변경하세요.
        #     만약 "fedCurv" 계열 아이디어처럼 Fisher를 축적해나가고 싶다면
        #     추가 로직을 여기에 넣어야 합니다.
        global_params = {name: param.data.clone() for name, param in self.rootModelStatic.named_parameters()}
        updated_fishers = self._average_fishers(self.clientsFisher, global_params)

        return self.resultRootModel, updated_fishers

    def registerFisher(self, path):
        """
        클라이언트에서 계산되어 서버로 업로드된 Fisher 정보를 불러오는 메서드.
        JSON 포맷으로 저장된 정보를 로드하여 tensor로 변환.
        """
        if path is None:
            return
        import json
        with open(path, 'r') as f:
            fisher_json = json.load(f)

        fisher = {name: torch.tensor(param, device=self.device)
                  for name, param in fisher_json.items()}
        self.clientsFisher.append(fisher)

    def _average_fishers(self, fishers: List[Dict[str, torch.Tensor]], params: Dict[str, torch.Tensor]):
        """
        예시) Fisher를 단순 평균내는 함수. 필요에 따라 다른 계산 방식을 사용할 수 있음.
        """
        if not fishers:
            return None

        aggregated_fisher = {name: torch.zeros_like(param) for name, param in params.items()}
        num_clients = len(fishers)

        for name in params.keys():
            for client_id in range(num_clients):
                aggregated_fisher[name] += fishers[client_id][name] / num_clients

        return aggregated_fisher
