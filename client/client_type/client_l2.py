import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from client.client_type.client_parent import client_parent
from client.util_client import target_type_convert, clip_implement

class client_l2(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        """
        target_layers: list of substrings. Only parameters whose names
                       contain one of these substrings will be used
                       in the L2 distance penalty step.
                       e.g. ["conv1", "fc"].
        """
        super().__init__(client_internalId, dataset, networkConfig, basicConfig,
                         clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                         wandbQueue)

        target_layers = self.config['reg_layer']
        self.target_layers = target_layers if target_layers is not None else []

    def train(self, epochs=10):
        lr_origin = self.clientProfile['clientMetadata']['lr']
        lr = lr_origin

        # L2 정규화 강도(λ)
        penalty_lambda = self.clientProfile['clientMetadata']['penalty_lambda']

        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # 전역(또는 이전 라운드) 모델 복사 -> L2 distance reference
        global_model = copy.deepcopy(self.modelReserved).to(self.device)
        global_model.eval()  # reference용으로 freeze

        all_targets = []
        all_outputs = []

        # ------------------------------------------
        # (1) Train Loop (두 번 업데이트)
        # ------------------------------------------
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()

                # 1) Forward + classification loss
                outputs = self.model(inputs)
                cls_loss = self.criterion(outputs, targets)

                if len(self.target_layers) > 0 and penalty_lambda > 0:
                    # 2) L2 거리 계산 (지정된 레이어만)
                    l2_dist = 0.0
                    for (name_local, w_local), (name_global, w_global) in zip(
                            self.model.named_parameters(),
                            global_model.named_parameters()
                    ):
                        if any(t_layer in name_local for t_layer in self.target_layers):
                            l2_dist += torch.sum((w_local - w_global) ** 2)

                    if l2_dist > 0:
                        l2_loss = (penalty_lambda / 2.0) * l2_dist
                        cls_loss += l2_loss

                # 2) Backprop
                cls_loss.backward()

                if self.config['normClip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['normClip'])

                self.optimizer.step()

                running_loss += cls_loss.item()
                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            # 에폭 종료 후 평균 Loss
            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

            # (옵션) wandbQueue 로깅
            # self.wandbQueue.put(logList)
            # print(f"Client {self.client_internalId} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return logList
