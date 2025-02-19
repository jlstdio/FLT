import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from client.client_type.client_parent import client_parent
from client.util_client import target_type_convert, clip_implement


class client_pearson(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        """
        target_layers: list of substrings used to match parameter names. Only
                       parameters whose name includes one of these substrings
                       will be used for the Pearson correlation regularization.
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

        # FedProx에서 사용하던 penalty_lambda를 Pearson 정규화 강도로 사용
        penalty_lambda = self.clientProfile['clientMetadata']['penalty_lambda']

        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # Reference model (e.g., global or previous round)
        prox_model = copy.deepcopy(self.modelReserved).to(self.device)
        prox_model.eval()

        all_targets = []
        all_outputs = []

        # -----------------------------------------------------------
        # (A) 모델 파라미터의 Pearson 상관계수(부분적) 계산 함수
        # -----------------------------------------------------------
        def param_pearson_correlation(model_a, model_b, target_layers, eps=1e-8):
            """
            Flattens only the parameters from 'target_layers' of model_a and model_b,
            then computes the Pearson correlation coefficient between them.
            eps is a small constant to avoid division by zero.
            """
            vecs_a = []
            vecs_b = []

            for (name_a, param_a), (name_b, param_b) in zip(
                model_a.named_parameters(), model_b.named_parameters()
            ):
                # If any substring in target_layers is part of name_a, include that parameter.
                if any(t_layer in name_a for t_layer in target_layers):
                    vecs_a.append(param_a.view(-1))
                    vecs_b.append(param_b.view(-1))

            # If no parameters match, return None
            if len(vecs_a) == 0:
                return None

            vec_a = torch.cat(vecs_a, dim=0)
            vec_b = torch.cat(vecs_b, dim=0)

            mean_a = vec_a.mean()
            mean_b = vec_b.mean()
            var_a = (vec_a - mean_a).pow(2).sum()
            var_b = (vec_b - mean_b).pow(2).sum()

            cov_ab = ((vec_a - mean_a) * (vec_b - mean_b)).sum()
            corr = cov_ab / (torch.sqrt(var_a * var_b) + eps)

            return corr

        # -----------------------------------------------------------
        # (B) Training Loop (두 번 업데이트)
        # -----------------------------------------------------------
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets)
                targets = targets.to(self.device)

                for param in self.model.parameters():
                    param.requires_grad = True  # 전체 레이어 학습 가능

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                cls_loss = self.criterion(outputs, targets)

                if len(self.target_layers) > 0 and penalty_lambda > 0:

                    pearson_corr = param_pearson_correlation(
                        self.model, prox_model, target_layers=self.target_layers
                    )
                    if pearson_corr is not None:
                        # 정규화 항: 1 - corr
                        pearson_reg = 1.0 - pearson_corr
                        reg_loss = penalty_lambda * pearson_reg
                        cls_loss += reg_loss

                cls_loss.backward()

                if self.config['normClip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['normClip'])

                self.optimizer.step()

                running_loss += cls_loss.item()
                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            # 에폭마다 평균 Loss 기록
            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

            # (옵션) wandbQueue 로깅 등
            # self.wandbQueue.put(logList)
            # print(f"Client {self.client_internalId} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return logList
