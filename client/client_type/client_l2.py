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

                # -----------------------------------------------------
                # [A] 첫 번째 업데이트: Classification Step (전체 레이어)
                # -----------------------------------------------------
                for param in self.model.parameters():
                    param.requires_grad = True  # 모든 레이어 unfreeze

                self.optimizer.zero_grad()

                # 1) Forward + classification loss
                outputs = self.model(inputs)
                cls_loss = self.criterion(outputs, targets)

                # 2) Backprop
                cls_loss.backward()
                clip_implement(self.config['costFunc'], self.model, self.config['normClip'])
                self.optimizer.step()

                running_loss += cls_loss.item()
                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

                # -----------------------------------------------------
                # [B] 두 번째 업데이트: Partial L2 Step (특정 레이어만)
                # -----------------------------------------------------
                if len(self.target_layers) > 0 and penalty_lambda > 0:
                    # 1) Freeze: target_layers가 아닌 레이어는 grad X
                    for name, param in self.model.named_parameters():
                        if any(t_sub in name for t_sub in self.target_layers):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                    self.optimizer.zero_grad()

                    # 2) L2 거리 계산 (지정된 레이어만)
                    l2_dist = 0.0
                    for (name_local, w_local), (name_global, w_global) in zip(
                        self.model.named_parameters(),
                        global_model.named_parameters()
                    ):
                        if any(t_layer in name_local for t_layer in self.target_layers):
                            l2_dist += torch.sum((w_local - w_global) ** 2)

                    # 3) 로스: penalty_lambda * (l2_dist / 2)
                    #    (스케일은 필요에 따라 조정 가능)
                    if l2_dist > 0:
                        l2_loss = (penalty_lambda / 2.0) * l2_dist
                        l2_loss.backward()
                        clip_implement(self.config['costFunc'], self.model, self.config['normClip'])
                        self.optimizer.step()

            # 에폭 종료 후 평균 Loss
            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

            # (옵션) wandbQueue 로깅
            # self.wandbQueue.put(logList)
            # print(f"Client {self.client_internalId} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return logList
