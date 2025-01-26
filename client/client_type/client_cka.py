import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from client.client_type.client_parent import client_parent
from client.util_client import target_type_convert, criterion_select, clip_implement

# --------------------------------------------------
# [1] (Same CKA utilities as before)
# --------------------------------------------------
def center_gram(K):
    n = K.size(0)
    I = torch.eye(n, device=K.device)
    ones = torch.ones((n, n), device=K.device) / n
    H = I - ones
    return H @ K @ H

def linear_cka(X, Y):
    N = X.size(0)
    X = X.view(N, -1)
    Y = Y.view(N, -1)

    Kx = X @ X.t()
    Ky = Y @ Y.t()

    Kx_centered = center_gram(Kx)
    Ky_centered = center_gram(Ky)

    hsic_xy = (Kx_centered * Ky_centered).sum() / ((N - 1) ** 2)
    hsic_xx = (Kx_centered * Kx_centered).sum() / ((N - 1) ** 2)
    hsic_yy = (Ky_centered * Ky_centered).sum() / ((N - 1) ** 2)

    return hsic_xy / torch.sqrt(hsic_xx * hsic_yy + 1e-8)

def cka_loss(X, Y):
    return 1.0 - linear_cka(X, Y)


# --------------------------------------------------
# [2] Forward hooks for capturing specific layers
# --------------------------------------------------
def register_hooks(model, layer_names):
    outputs_dict = {}

    def make_hook(layer_name):
        def hook(module, inp, out):
            outputs_dict[layer_name] = out
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    return outputs_dict, hooks

def remove_hooks(hooks):
    for h in hooks:
        h.remove()


class client_cka(client_parent):
    def __init__(self, client_internalId, dataset, networkConfig, basicConfig,
                 clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                 wandbQueue):
        super().__init__(client_internalId, dataset, networkConfig, basicConfig,
                         clientType, config, model, serverRound, flipboard, turnFlag, sessionId, scorePath,
                         wandbQueue)

        # reg_layer에 지정된 레이어 이름 리스트
        cka_layers = self.config['reg_layer']
        self.cka_layers = cka_layers if cka_layers is not None else []

    def train(self, epochs=10):
        lr_origin = self.clientProfile['clientMetadata']['lr']
        lr = lr_origin

        # penalty_lambda가 CKA 정규화 계수로 사용됨
        penalty_lambda = self.clientProfile['clientMetadata']['penalty_lambda']

        logList = None
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # 이전(글로벌) 모델 복사 -> prox_model (CKA 비교용)
        prox_model = copy.deepcopy(self.modelReserved).to(self.device)
        prox_model.eval()  # Reference model은 고정

        all_targets = []
        all_outputs = []

        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0

            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets)
                targets = targets.to(self.device)

                ###
                # (1) Classification Update: 모든 레이어에 대해서 unfreeze -> 한 번의 업데이트
                ###
                for param in self.model.parameters():
                    param.requires_grad = True

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                cls_loss = self.criterion(outputs, targets)

                cls_loss.backward()
                clip_implement(self.config['costFunc'], self.model, self.config['normClip'])
                self.optimizer.step()

                running_loss += cls_loss.item()

                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

                ###
                # (2) [선택사항] CKA Update: cka_layers가 있으면, 해당 레이어만 unfreeze하여 업데이트
                ###
                if len(self.cka_layers) > 0 and penalty_lambda > 0:
                    # ---- Freeze / Unfreeze 설정 ----
                    for name, param in self.model.named_parameters():
                        if any(layer_name in name for layer_name in self.cka_layers):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                    self.optimizer.zero_grad()

                    # Hook 등록
                    current_outputs_dict, current_hooks = register_hooks(self.model, self.cka_layers)
                    prox_outputs_dict, prox_hooks = register_hooks(prox_model, self.cka_layers)

                    # Forward pass (현재 모델)
                    _ = self.model(inputs)
                    # Forward pass (prox_model)
                    with torch.no_grad():
                        _ = prox_model(inputs)

                    # CKA term 계산
                    cka_term_total = 0.0
                    for layer_name in self.cka_layers:
                        cur_feat = current_outputs_dict[layer_name]
                        ref_feat = prox_outputs_dict[layer_name]
                        cka_term_total += cka_loss(cur_feat, ref_feat)

                    cka_term_avg = cka_term_total / len(self.cka_layers)
                    cka_loss_value = penalty_lambda * cka_term_avg

                    cka_loss_value.backward()
                    clip_implement(self.config['costFunc'], self.model, self.config['normClip'])
                    self.optimizer.step()

                    # Hook 해제
                    remove_hooks(current_hooks)
                    remove_hooks(prox_hooks)

            # 에폭 완료 후 평균 Loss
            avg_loss = running_loss / len(self.train_loader)

            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

            # (예시) wandb 로깅 가능
            # self.wandbQueue.put(logList)
            # self.wandbClient.sendLog(key=..., data=...)

            # print(f"Client {self.client_internalId} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return logList
