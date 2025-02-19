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
def check_nan_inf(tensor, name, fallback=0.0):
    """tensor의 NaN/Inf 여부를 검사하여, 발견 시 경고 메시지 출력 후 True 반환"""
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN")
        return True
    if torch.isinf(tensor).any():
        print(f"Warning: {name} contains Inf")
        return True
    return False


def linear_cka(X, Y, eps=1e-8, fallback=0.0):
    N = X.size(0)
    X = X.view(N, -1)
    Y = Y.view(N, -1)

    # Gram matrix 계산
    Kx = X @ X.t()
    Ky = Y @ Y.t()

    # Gram matrix를 센터링
    Kx_centered = center_gram(Kx)
    Ky_centered = center_gram(Ky)

    # HSIC 값 계산
    hsic_xy = (Kx_centered * Ky_centered).sum() / ((N - 1) ** 2)
    hsic_xx = (Kx_centered * Kx_centered).sum() / ((N - 1) ** 2)
    hsic_yy = (Ky_centered * Ky_centered).sum() / ((N - 1) ** 2)

    # 클램핑(0 이하로 내려가지 않도록)
    hsic_xx = torch.clamp(hsic_xx, min=eps)
    hsic_yy = torch.clamp(hsic_yy, min=eps)

    # CKA 값 계산
    cka_val = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    if check_nan_inf(cka_val, "cka_val", fallback):
        print("Warning: CKA computed as NaN or Inf")
        return fallback

    return cka_val


def center_gram(K):
    """
    예시용 center_gram 함수.
    Gram matrix K를 double-centering 하는 기능을 수행.
    (아래는 예시 구현이므로 실제 쓰이는 로직과 맞추어 수정하세요.)
    """
    n = K.size(0)
    I = torch.eye(n, device=K.device)
    ones = torch.ones((n, n), device=K.device) / n
    K_centered = K - ones @ K - K @ ones + ones @ K @ ones
    return K_centered


def cka_loss(X, Y):
    # return 1.0 - linear_cka(X, Y)
    return linear_cka(X, Y)


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

                if len(self.cka_layers) > 0 and penalty_lambda > 0 and inputs.size(0) > 1:
                    current_outputs_dict, current_hooks = register_hooks(self.model, self.cka_layers)
                    prox_outputs_dict, prox_hooks = register_hooks(prox_model, self.cka_layers)

                inputs = inputs.to(self.device)
                targets = target_type_convert(self.config['costFunc'], targets)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if len(self.cka_layers) > 0 and penalty_lambda > 0 and inputs.size(0) > 1:

                    with torch.no_grad():
                        _ = prox_model(inputs)

                    # CKA term 계산
                    cka_term_total = 0.0
                    cka_term_min = 1e-8
                    cka_term_max = 1.0
                    for layer_name in self.cka_layers:
                        cur_feat = current_outputs_dict[layer_name]
                        ref_feat = prox_outputs_dict[layer_name]
                        cka_loss_value = cka_loss(cur_feat, ref_feat)
                        cka_loss_value = torch.clamp(cka_loss_value, cka_term_min, cka_term_max)
                        cka_term_total += cka_loss_value

                    cka_term_avg = cka_term_total / len(self.cka_layers)
                    cka_loss_value = penalty_lambda * cka_term_avg

                    loss += cka_loss_value

                    # Hook 해제
                    remove_hooks(current_hooks)
                    remove_hooks(prox_hooks)

                loss.backward()

                if self.config['normClip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['normClip'])

                self.optimizer.step()
                running_loss += loss.item()

                all_targets.extend(targets.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

            # 에폭 완료 후 평균 Loss
            avg_loss = running_loss / len(self.train_loader)

            key_loss = f"client/performance/train/loss/client{self.client_internalId} training loss"
            logList = [key_loss, avg_loss, self.round]

            # (예시) wandb 로깅 가능
            # self.wandbQueue.put(logList)
            # self.wandbClient.sendLog(key=..., data=...)

            # print(f"Client {self.client_internalId} Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return logList
