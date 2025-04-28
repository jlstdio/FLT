import os, copy, json, time, glob, math
from typing import List, Dict, Tuple
import numpy as np
import torch, torch.nn as nn
from sklearn.decomposition import PCA
from util.data_prepare.data_prepare_manager import select_dataset

dataset_for_type = {
        0: "cifar-10_jg",
        1: "cifar-10_jo",
        2: "cifar-10_jp",
        3: "cifar-10_r1",
        4: "cifar-10_r2",
        5: "cifar-10_r3",
        6: "cifar-10_r4",
        7: "cifar-10_lp",
        8: "cifar-10_bp",
        9: "cifar-10_bs",
}
style_list = [dataset_for_type[i] for i in sorted(dataset_for_type)]         # 10 종

def load_merged_and_replicate(style_list, per_class=10):
    all_data_sub = []
    ctr = {c: 0 for c in range(10)}
    for name in style_list:
        _, test_ds, _ = select_dataset(name)
        for item in test_ds:
            lbl = item[0]
            if ctr[lbl] < per_class:
                all_data_sub.append(item)
                ctr[lbl] += 1
            if all(v == per_class for v in ctr.values()):
                break
        ctr = {c: 0 for c in range(10)}
    style_datasets = [all_data_sub for _ in range(10)]      # 각 클라이언트에 동일 데이터
    return style_datasets                                   # 길이 10, 각 100 샘플

STYLE_DATASETS = load_merged_and_replicate(style_list)      # 전역 캐싱

# ──────────────────────────────────────────────────────────────
# 1. 모델 구조 (루트/클라이언트와 동일해야 함)
# ──────────────────────────────────────────────────────────────
class testNN_wo_Softmax_3_layer(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.fc    = nn.Linear(64 * 8 * 8, out_classes)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = self.relu(self.conv3(x))
        return self.fc(x.view(x.size(0), -1))

# ──────────────────────────────────────────────────────────────
# 2. 레이어별 forward (activation 추출용)
# ──────────────────────────────────────────────────────────────
def _layer_acts(model, imgs, layer):
    r = nn.ReLU(inplace=False)
    if layer == "conv1":
        x = r(model.pool(model.conv1(imgs)))
    elif layer == "conv2":
        x = r(model.pool(model.conv1(imgs)))
        x = r(model.pool(model.conv2(x)))
    elif layer == "conv3":
        x = r(model.pool(model.conv1(imgs)))
        x = r(model.pool(model.conv2(x)))
        x = r(model.conv3(x))
    elif layer == "fc":
        x = r(model.pool(model.conv1(imgs)))
        x = r(model.pool(model.conv2(x)))
        x = r(model.conv3(x))
        x = model.fc(x.view(x.size(0), -1))
    else:
        raise ValueError(layer)
    return x.view(x.size(0), -1).detach()

# ──────────────────────────────────────────────────────────────
# 3. Ω 계산 (사용자 제공 코드를 그대로 옮김)
# ──────────────────────────────────────────────────────────────
def greedy_feature_clustering(corr, gamma):
    N = corr.shape[0]
    labels  = np.full(N, -1, int)
    visited = np.zeros(N, bool)
    cid = 0
    for i in range(N):
        if visited[i]: continue
        stack = [i]; visited[i] = True; labels[i] = cid
        while stack:
            u = stack.pop()
            for v in np.where(np.abs(corr[u]) >= gamma)[0]:
                if not visited[v]:
                    visited[v] = True; labels[v] = cid; stack.append(v)
        cid += 1
    return labels, cid

def compute_interaction_tensor_multi(models, style_datasets,
                                     device="cpu", layer="conv1",
                                     k_pca=50, thresh=90):
    M, k = len(models), k_pca
    rows, sizes = [], []
    for m, ds in zip(models, style_datasets):
        imgs = torch.stack([torch.tensor(img.transpose(2,0,1)) for _, img in ds]).to(device).float()
        A = _layer_acts(m.to(device), imgs, layer).cpu().numpy()      # (N,D)
        rows.append(PCA(n_components=k).fit_transform(A).T)           # (k,N)
        sizes.append(len(ds))
    X = np.concatenate(rows, axis=0)
    X -= X.mean(1, keepdims=True)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    corr = Xn @ Xn.T
    gamma_corr = thresh / 100.0
    labels, T = greedy_feature_clustering(corr, gamma_corr)
    Omega = np.zeros((M, max(sizes), T), np.int8)
    row_ptr = 0
    for m_idx, N in enumerate(sizes):
        for i in range(k):
            row = row_ptr + i
            cid = labels[row]
            gamma_data = np.percentile(np.abs(Xn[row]), thresh)
            mask = np.abs(Xn[row]) >= gamma_data
            Omega[m_idx, mask[:N], cid] = 1
        row_ptr += k
    return Omega                                                     # (M,N,T)

# ──────────────────────────────────────────────────────────────
# 4. 중요도 α 계산 및 정규화
# ──────────────────────────────────────────────────────────────
def omega_to_alpha(omega_m):
    return omega_m.mean(0)                                           # (T,)

def normalize_alphas(raw_list):
    Tmax = max(x.size for x in raw_list)
    mat  = np.vstack([np.pad(x, (0, Tmax - x.size)) for x in raw_list])
    mat  = mat / (mat.sum(0, keepdims=True) + 1e-12)
    return [torch.tensor(v, dtype=torch.float32) for v in mat]

# ──────────────────────────────────────────────────────────────
# 5. 필터별 가중 평균
# ──────────────────────────────────────────────────────────────
def wavg_conv(stacked, w):
    w = w.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(stacked)
    return (w*stacked).sum(0) / (w.sum(0)+1e-12)

def wavg_fc(stacked, w):
    w = w.unsqueeze(2).expand_as(stacked)
    return (w*stacked).sum(0) / (w.sum(0)+1e-12)

def wavg_bias(stacked, w):
    return (w*stacked).sum(0) / (w.sum(0)+1e-12)

# ──────────────────────────────────────────────────────────────
# 6. 부모 클래스 대체 (프로젝트側 fedOptParent 인터페이스 최소 구현)
# ──────────────────────────────────────────────────────────────
class _MiniParent:
    def __init__(self, rootModel, cudaId):
        self.resultRootModel = copy.deepcopy(rootModel)
        self.device = torch.device(f"cuda:{cudaId}" if torch.cuda.is_available() else "cpu")
        self.cudaId = cudaId
        self.clientsModels = []

    def flush(self):                # server_feature_wise 에서 호출
        self.clientsModels.clear()

    def registerPth(self, path):    # pth 파일을 읽어 state_dict 리스트에 적재
        sd = torch.load(path, map_location="cpu")
        self.clientsModels.append(sd)

    def registerFisher(self, _):    # 호환용 더미
        pass

    def afterWork(self):            # 호환용 더미
        pass

# ──────────────────────────────────────────────────────────────
# 7. 최종 Aggregator
# ──────────────────────────────────────────────────────────────
class fed_feature_wise_weighted_avg(_MiniParent):
    def __init__(self, rootModel, cudaId, *_):
        super().__init__(rootModel, cudaId)

    # ------------------------------------------------------ #
    def _calc_omegas_alphas(self, models):
        layers = ["conv1", "conv2", "conv3", "fc"]
        omegas, alphas = {}, {}
        for ℓ in layers:
            Ω = compute_interaction_tensor_multi(models,
                                                 STYLE_DATASETS,
                                                 device=self.device,
                                                 layer=ℓ,
                                                 k_pca=50, thresh=90)
            omegas[ℓ] = Ω
            alphas_raw = [omega_to_alpha(Ω[m]) for m in range(Ω.shape[0])]
            alphas[ℓ] = normalize_alphas(alphas_raw)               # torch 리스트
        return alphas                                              # {ℓ:[α1…]}

    # ------------------------------------------------------ #
    def aggregate(self):
        assert self.clientsModels, "[FW-Weight] no client models"

        # state_dict → nn.Module 복원
        models = []
        for sd in self.clientsModels:
            m = testNN_wo_Softmax_3_layer(10).to(self.device)
            m.load_state_dict(sd, strict=True)
            models.append(m)

        alphas = self._calc_omegas_alphas(models)

        # 클라이언트 state_dict stack
        keys = self.clientsModels[0].keys()
        new_sd = {}
        for k in keys:
            stacked = torch.stack([c[k] for c in self.clientsModels], 0).to(self.device)

            if   "conv1.weight" in k:
                w = torch.stack(alphas["conv1"]).to(self.device)     # (M,32)
                new_sd[k] = wavg_conv(stacked, w)
            elif "conv2.weight" in k:
                w = torch.stack(alphas["conv2"]).to(self.device)
                new_sd[k] = wavg_conv(stacked, w)
            elif "conv3.weight" in k:
                w = torch.stack(alphas["conv3"]).to(self.device)
                new_sd[k] = wavg_conv(stacked, w)
            elif "fc.weight" in k:
                w = torch.stack(alphas["fc"]).to(self.device)        # (M,10)
                new_sd[k] = wavg_fc(stacked, w)
            elif ".bias" in k:
                if   "conv1" in k: w = torch.stack(alphas["conv1"])
                elif "conv2" in k: w = torch.stack(alphas["conv2"])
                elif "conv3" in k: w = torch.stack(alphas["conv3"])
                else:              w = torch.stack(alphas["fc"])
                new_sd[k] = wavg_bias(stacked, w.to(self.device))
            else:                              # BN 등은 단순 평균
                new_sd[k] = stacked.mean(0)

        self.resultRootModel.load_state_dict(new_sd, strict=True)
        return self.resultRootModel
