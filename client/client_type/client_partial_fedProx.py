import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from client.client_type.client_parent import client_parent
from client.util_client import target_type_convert, criterion_select, clip_implement


class client_partial_fedprox(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        super().__init__(client_internalId, dataset, networkConfig, basicConfig,
                         clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                         wandbQueue)

        # FedProx에서 Proximal Term을 적용할 레이어 이름 리스트
        # (기존 코드에서 self.config['reg_layer']를 cka_layers로 사용했으므로 그대로 둠)
        cka_layers = self.config['reg_layer']
        self.cka_layers = cka_layers if cka_layers is not None else []

    def train(self, epochs=10):
        # 기본 학습 설정
        lr_origin = self.clientProfile['clientMetadata']['lr']
        lr = lr_origin

        # FedProx에서 사용하는 계수(여기서는 penalty_lambda로 명시)
        penalty_lambda = self.clientProfile['clientMetadata']['penalty_lambda']

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.model.train()

        # ---------------------------
        # (1) 글로벌(이전) 모델 복제
        # ---------------------------
        prox_model = copy.deepcopy(self.modelReserved).to(self.device)
        prox_model.eval()  # reference model은 고정

        # 학습 로그 (예시)
        logList = None

        all_targets = []
        all_outputs = []

        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets)
                targets = targets.to(self.device)

                # Forward
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # --------------------------------------------------
                # (2) FedProx Proximal Term (지정된 레이어만 계산)
                # --------------------------------------------------
                if len(self.cka_layers) > 0 and penalty_lambda > 0:
                    prox_term = 0.0

                    # model과 prox_model의 각 파라미터를 순회하며
                    # 해당 파라미터 이름(name)이 self.cka_layers 중 하나를 포함하면 Prox Term에 더함
                    for (name, param), (_, param_ref) in zip(self.model.named_parameters(),
                                                             prox_model.named_parameters()):
                        # 지정된 레이어 이름이 name에 포함되면 해당 파라미터만 계산
                        if any(layer_name in name for layer_name in self.cka_layers):
                            prox_term += torch.sum((param - param_ref) ** 2)

                    if prox_term > 0:
                        # FedProx에서는 보통 (mu / 2) * ||w - w_ref||^2 형태로 사용
                        prox_loss = 0.5 * penalty_lambda * prox_term
                        loss += prox_loss

                # Backward & step
                loss.backward()

                if self.config['normClip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['normClip'])

                self.optimizer.step()
                running_loss += loss.item()

                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            # Epoch 별 평균 Loss 계산
            avg_loss = running_loss / len(self.train_loader)

            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

            # wandb나 기타 로깅이 필요하다면 여기에 추가
            # self.wandbQueue.put(logList)
            # 또는
            # self.wandbClient.sendLog(key=..., data=...)

            # print(f"Client {self.client_internalId} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return logList
