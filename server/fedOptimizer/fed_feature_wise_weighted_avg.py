import os, glob, math, copy, numpy as np, torch
from torch import nn
from sklearn.decomposition import PCA
from numba.cuda import is_available
from typing import Dict, List, Callable
from server.fedOptimizer.fedOptParent import fedOptParent
from util.util import loadData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def act_conv1(m, x):
    return m.relu(m.pool(m.conv1(x))).view(x.size(0),-1)
def act_conv2(m,x):
    a = m.relu(m.pool(m.conv1(x)))
    return m.relu(m.pool(m.conv2(a))).view(x.size(0),-1)
def act_conv3(m,x):
    a = m.relu(m.pool(m.conv1(x)))
    a = m.relu(m.pool(m.conv2(a)))
    return m.relu(m.conv3(a)).view(x.size(0),-1)

LAYER_FNS:Dict[str,Callable] = {
    'conv1': act_conv1,
    'conv2': act_conv2,
    'conv3': act_conv3     # penultimate
}


def get_activation_for_ds(model, ds, act_fn, device="cpu"):
    embs = []
    model.eval()
    for _, img in ds:
        if isinstance(img, np.ndarray):
            # print("img shape", img.shape) # img shape (32, 32, 3)
            img_t = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)
            # print("img shape", img_t.shape) # img shape torch.Size([1, 3, 32, 32])
        else:
            img_t = img.unsqueeze(0)
        
        embs.append(act_fn(model, img_t.to(device)).detach().cpu().numpy())
    
    print(f"get_activation_for_ds: {len(embs)} images, {embs[0].shape} activations")
    return np.vstack(embs)


def greedy_feature_clustering(corr: np.ndarray, gamma: float):
    N = corr.shape[0]
    lab = -np.ones(N, int); vis=np.zeros(N, bool); cid=0
    for i in range(N):
        if vis[i]: continue
        stack=[i]; vis[i]=True; lab[i]=cid
        while stack:
            u=stack.pop()
            for v in np.where(np.abs(corr[u])>=gamma)[0]:
                if not vis[v]:
                    vis[v]=True; lab[v]=cid; stack.append(v)
        cid+=1
    return lab, cid


def layer_interaction_tensor(models, act_fn, style_datasets, device=device, k_pca=50, thresh=99.5):
    M, k = len(models), k_pca
    rows = []
    # rows = pca maps

    print(f'[TEST] models: {len(models)}')
    for idx, m in enumerate(models):
        print(f"{idx} model")
        A = get_activation_for_ds(m, copy.deepcopy(style_datasets), act_fn, device) # (N_style × D)
        P = PCA(n_components=k).fit_transform(A).T # (k × N_style)
        rows.append(P)
        
    X = np.concatenate(rows, axis=0) # (M*k) × N_total
    X -= X.mean(1, keepdims=True)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    corr = Xn @ Xn.T
    tril = np.tril_indices(corr.shape[0], -1)
    # gamma_corr = np.percentile(np.abs(corr[tril]), thresh)
    gamma_corr = thresh
    gamma_data = np.percentile(np.abs(Xn), thresh)

    labels, T = greedy_feature_clustering(corr, gamma_corr)
    Omega = np.zeros((M, Xn.shape[1], T), np.int8)
    
    for m_idx in range(M):
        for i in range(k):
            row = m_idx * k + i
            cid = labels[row]
            Omega[m_idx, np.where(np.abs(Xn[row]) >= gamma_data)[0], cid] = 1
    return torch.from_numpy(Omega), torch.from_numpy(rows)


def _grad_importance(model, imgs, pca_vec, act_fn):
    model.zero_grad(set_to_none=True)
    imgs = imgs.to(device).requires_grad_(True)

    rep = act_fn(model, imgs) # (B,D)
    score = (rep @ pca_vec.to(device)).sum() # scalar s
    grads = torch.autograd.grad(score, model.parameters(), retain_graph=False, allow_unused=True)
    return {pname: (g.abs().mean()
            if g is not None else torch.tensor(0.,device=device))
            for (pname,_),g in zip(model.named_parameters(),grads)}


def grad_importance_map(models: List[nn.Module], 
                        pca_mat: List[torch.Tensor], # 각 모델별 PCA 결과 (k × N)
                        Omega: torch.Tensor, # shape = (M, N, F)
                        dataset, act_fn, top_freq=0.0):

    M, N, F = Omega.shape

    print(f"grad_importance_map: {M} models, {N} data, {F} features")

    # 🔸 1. 모델별 feature 빈도 계산 및 정규화
    g = Omega.sum(dim=1)  # (M, F)
    g = g / (g.max(dim=1, keepdim=True).values + 1e-12)
    print("g shape: ", g.shape)

    # 🔸 2. threshold 계산 (옵션)
    if top_freq > 0:
        thresh = torch.quantile(g, 1 - top_freq, dim=1, keepdim=True)  # 모델마다 다른 임계값

    imap = {n: torch.tensor(0.0, device=next(models[0].parameters()).device) 
            for n, _ in models[0].named_parameters()}

    # 🔸 3. 모델마다 순회
    for m, model in enumerate(models):
        model.eval()
        for t in range(F):
            if g[m, t] == 0: continue
            if top_freq > 0 and g[m, t] < thresh[m]: continue

            pvec = pca_mat[m][t].to(next(model.parameters()).device)  # shape: (k,) (1D PCA vector)
            
            for _, img in dataset:
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
                    #img_t = torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0)
                else:
                    img = img.unsqueeze(0).float()
                
                grads = _grad_importance(model, img.to(pvec.device), pvec, act_fn)
                for k, v in grads.items():
                    imap[k] += g[m, t] * v

    # 🔸 4. 정규화
    s = sum(imap.values()) + 1e-12
    for k in imap:
        imap[k] /= s

    return imap


def weighted_avg_param(param_name, client_states, wmaps):
    stk = torch.stack([cs[param_name] for cs in client_states],0).float()
    w   = torch.tensor([wm[param_name] for wm in wmaps],
                       dtype=stk.dtype,device=stk.device).view(-1,*[1]*(stk.ndim-1))
    return (w*stk).sum(0)/(w.sum()+1e-9)

class fed_feature_wise_weighted_avg(fedOptParent):

    def __init__(self, rootModel, cudaId, additionalInfo=None):

        self.additionalInfo = additionalInfo
        self.rootModelStatic = copy.deepcopy(rootModel)
        self.resultRootModel = copy.deepcopy(self.rootModelStatic)
        self.clientsModels = []
        self.clientsLosses = []
        self.clients_types = []
        self.clients_ids = []
        self.max_retries = 10
        self.device = torch.device(f"cuda:{cudaId}" if is_available() else "cpu")

        self.merged_data = self.additionalInfo['dataset']
        self.layer_datasets = {l: self.merged_data for l in LAYER_FNS}

    def aggregate(self, k_pca=50):
        client_states = self.clientsModels
        M = len(client_states)

        # 레이어별 클라이언트 weight-map
        layer_wmaps={l:[] for l in LAYER_FNS}

        client_models = []
        
        # client별 추론 모델 생성
        for m_idx, state in enumerate(client_states):
            model = copy.deepcopy(self.resultRootModel).to(device)
            model.load_state_dict(state)  # Remove assignment since load_state_dict() returns _IncompatibleKeys
            client_models.append(copy.deepcopy(model))

        for l, act_fn in LAYER_FNS.items():
            print(f"Layer {l} layer")
            Omega, P = layer_interaction_tensor(client_models, act_fn, self.layer_datasets[l], k_pca=k_pca, thresh=99.5)
            wm = grad_importance_map(client_models, P, Omega, self.layer_datasets[l], act_fn)
            layer_wmaps[l].append(wm)
            

        # 레이어별 합치기
        new_state={}
        for l in LAYER_FNS:
            keys=[k for k in client_states[0].keys() if k.startswith(l)]
            for k in keys:
                new_state[k]=weighted_avg_param(k, client_states, layer_wmaps[l])

        self.resultRootModel.load_state_dict(new_state, strict=False)
        return self.resultRootModel