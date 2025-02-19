import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from client.client_type.client_parent import client_parent
from client.util_client import target_type_convert, clip_implement


class client_cosine(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        """
        target_layers: list of layer name substrings. Only parameters whose
                       name contains one of these substrings will be used in
                       the cosine similarity regularization step.
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

        # FedProx에서 사용하던 lambda를 여기서도 사용
        penalty_lambda = self.clientProfile['clientMetadata']['penalty_lambda']

        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # 글로벌(혹은 이전) 모델 복사 -> param-level Cosine 기준
        prox_model = copy.deepcopy(self.modelReserved).to(self.device)
        prox_model.eval()

        all_targets = []
        all_outputs = []

        # ----------------------------------------
        # [1] Parameter-level Cosine Similarity 함수
        # ----------------------------------------
        def param_cosine_similarity(modelA, modelB, target_layers):
            """
            modelA와 modelB에서 'target_layers'에 해당하는 파라미터만 추출,
            1D로 펼쳐 concat 후 cosine similarity 계산.
            """
            paramsA = []
            paramsB = []

            for (nameA, paramA), (nameB, paramB) in zip(modelA.named_parameters(),
                                                        modelB.named_parameters()):
                # 파라미터 이름에 target_layers 중 하나라도 포함되어 있으면 사용
                if any(layer_name in nameA for layer_name in target_layers):
                    paramsA.append(paramA.view(-1))
                    paramsB.append(paramB.view(-1))

            if len(paramsA) == 0:
                return None  # 매칭 파라미터가 없다면 None 반환

            vecA = torch.cat(paramsA, dim=0)
            vecB = torch.cat(paramsB, dim=0)

            # 1D 벡터 간 cosine similarity
            cos_sim = F.cosine_similarity(vecA, vecB, dim=0)
            return cos_sim

        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                cls_loss = self.criterion(outputs, targets)  # 분류 로스 예: cross-entropy

                if len(self.target_layers) > 0 and penalty_lambda > 0:
                    cos_sim = param_cosine_similarity(self.model, prox_model, self.target_layers)
                    if cos_sim is not None:
                        cos_loss = (1.0 - cos_sim) * penalty_lambda
                        cls_loss += cos_loss

                cls_loss.backward()

                if self.config['normClip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['normClip'])

                self.optimizer.step()
                running_loss += cls_loss.item()

                # 기록용
                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            avg_loss = running_loss / len(self.train_loader)
            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

            # (옵션) wandbQueue 등 로깅
            # self.wandbQueue.put(logList)
            # print(f"Client {self.client_internalId} Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        return logList
