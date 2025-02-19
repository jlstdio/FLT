import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from client.client_type.client_parent import client_parent
from client.util_client import target_type_convert, clip_implement


class client_em(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        """
        target_layers: list of layer name substrings to which we'll apply
                       partial L2 distance penalty in the second step.
                       e.g. ["conv1", "fc"].
        """
        super().__init__(client_internalId, dataset, networkConfig, basicConfig,
                         clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                         wandbQueue)

        target_layers = self.config['reg_layer']
        self.target_layers = target_layers if target_layers is not None else []

    def train(self, epochs=10):
        """
        [개요]
        1) 각 배치마다 EM 알고리즘의 E-step과 M-step을 통해 학습 (첫 번째 업데이트),
        2) 이어서 특정 레이어에만 partial L2 distance 규제를 적용 (두 번째 업데이트).
        """
        lr_origin = self.clientProfile['clientMetadata']['lr']
        lr = lr_origin

        # partial L2 규제 강도
        penalty_lambda = self.clientProfile['clientMetadata']['penalty_lambda']

        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # 글로벌/이전 모델 복사본 (Partial L2 비교용)
        prox_model = copy.deepcopy(self.modelReserved).to(self.device)
        prox_model.eval()

        all_targets = []
        all_outputs = []

        # ----------------------------
        # 0) Soft-label loss 함수
        # ----------------------------
        def soft_label_loss(logits, soft_labels):
            """
            logits: (N, num_classes)
            soft_labels: (N, num_classes) -> 각 샘플별 확률분포

            -sum(soft_labels * log_softmax(logits)) 형태의 cross entropy 유사 손실
            """
            log_probs = F.log_softmax(logits, dim=-1)
            loss = - (soft_labels * log_probs).sum(dim=-1).mean()
            return loss

        # -----------------------------------------------
        # Partial L2 distance (특정 레이어만)
        # -----------------------------------------------
        def partial_l2_distance(model_local, model_ref, target_layers_list):
            """
            Sums (w_local - w_ref)^2 over only those parameters
            whose names match any substring in target_layers_list.
            """
            l2_sum = 0.0
            for (name_local, param_local), (name_ref, param_ref) in zip(
                model_local.named_parameters(), model_ref.named_parameters()
            ):
                if any(t_sub in name_local for t_sub in target_layers_list):
                    l2_sum += torch.sum((param_local - param_ref) ** 2)
            return l2_sum

        # -----------------------------------------------
        # EM training (두 번의 update) Loop
        # -----------------------------------------------
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets)
                targets = targets.to(self.device)

                # ---------------------------------------------------
                # (1) E-step + M-step (Classification Update)
                #     모든 레이어 업데이트
                # ---------------------------------------------------
                # (A) E-step: 현 모델로 soft label 추론 (gradient X)
                with torch.no_grad():
                    logits = self.model(inputs)
                    soft_probs = F.softmax(logits, dim=-1)

                self.optimizer.zero_grad()
                new_logits = self.model(inputs)
                cls_loss = soft_label_loss(new_logits, soft_probs)

                if len(self.target_layers) > 0 and penalty_lambda > 0:
                    l2_dist = partial_l2_distance(self.model, prox_model, self.target_layers)
                    if l2_dist > 0:
                        l2_loss = (penalty_lambda / 2.0) * l2_dist
                        cls_loss += l2_loss

                cls_loss.backward()

                if self.config['normClip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['normClip'])

                self.optimizer.step()

                running_loss += cls_loss.item()

                # 기록용
                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(new_logits.detach().cpu().numpy())

            # 에폭별 평균 Loss (여기서는 EM classification 손실만 집계)
            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

            # print(f"Client {self.client_internalId} | Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            # (옵션) wandbQueue 로깅
            # self.wandbQueue.put(logList)

        return logList
