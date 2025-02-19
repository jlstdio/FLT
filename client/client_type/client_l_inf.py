import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from client.client_type.client_parent import client_parent
from client.util_client import target_type_convert, clip_implement


class client_l_inf(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        """
        target_layers: list of substrings. Only parameters whose names contain
                       one of these substrings will be included in the
                       L∞ distance penalty step. e.g. ["conv1", "fc"].
        """
        super().__init__(client_internalId, dataset, networkConfig, basicConfig,
                         clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                         wandbQueue)

        target_layers = self.config['reg_layer']
        self.target_layers = target_layers if target_layers is not None else []

    def train(self, epochs=10):
        lr_origin = self.clientProfile['clientMetadata']['lr']
        lr = lr_origin

        # 정규화 강도 (FedProx에서 사용하던 penalty_lambda 활용)
        penalty_lambda = self.clientProfile['clientMetadata']['penalty_lambda']

        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # 전역(또는 이전 라운드) 모델 복사 (L-inf distance 비교 대상)
        prox_model = copy.deepcopy(self.modelReserved).to(self.device)
        prox_model.eval()

        all_targets = []
        all_outputs = []

        # -------------------------------------------------
        # 1) L-inf Distance 계산 (특정 Layer만)
        # -------------------------------------------------
        def l_inf_distance(model_a, model_b, target_layers):
            """
            Collects only those parameters from model_a and model_b
            whose names match any substring in target_layers,
            then computes the L∞ norm (max absolute difference).
            """
            vecs_a = []
            vecs_b = []

            for (name_a, param_a), (name_b, param_b) in zip(
                model_a.named_parameters(), model_b.named_parameters()
            ):
                # If any target_layer substring is found in the param name, include it
                if any(t_layer in name_a for t_layer in target_layers):
                    vecs_a.append(param_a.view(-1))
                    vecs_b.append(param_b.view(-1))

            if len(vecs_a) == 0:  # 매칭되는 레이어가 없으면 0
                return 0.0

            flat_a = torch.cat(vecs_a, dim=0)
            flat_b = torch.cat(vecs_b, dim=0)

            # L-inf norm = max(|flat_a - flat_b|)
            dist = torch.norm(flat_a - flat_b, p=float("inf"))
            return dist

        # -------------------------------------------------
        # 2) Training Loop (두 번 업데이트)
        # -------------------------------------------------
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # 분류 로스
                outputs = self.model(inputs)
                cls_loss = self.criterion(outputs, targets)

                if len(self.target_layers) > 0 and penalty_lambda > 0:
                    l_inf_dist = l_inf_distance(self.model, prox_model, self.target_layers)

                    if l_inf_dist > 0:
                        reg_loss = penalty_lambda * l_inf_dist
                        cls_loss += reg_loss

                cls_loss.backward()

                if self.config['normClip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['normClip'])

                self.optimizer.step()

                running_loss += cls_loss.item()
                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            # 한 epoch 끝난 후 평균 loss
            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

            # (옵션) wandbQueue 로깅 등
            # self.wandbQueue.put(logList)
            # print(f"Client {self.client_internalId} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return logList
