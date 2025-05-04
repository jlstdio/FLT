from server.fedOptimizer.fedOptParent import fedOptParent
import os, copy, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from numba.cuda import is_available
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
import numba
torch.multiprocessing.set_sharing_strategy("file_system")# FD ↓

# ─────────── activation funcs (batch) ────────────────────────────────
def act_conv1_batch(m, x): return m.relu(m.pool(m.conv1(x))).flatten(1)
def act_conv2_batch(m, x): x = m.relu(m.pool(m.conv1(x))); return m.relu(m.pool(m.conv2(x))).flatten(1)
def act_conv3_batch(m, x): x = m.relu(m.pool(m.conv1(x))); x = m.relu(m.pool(m.conv2(x))); return m.relu(m.conv3(x)).flatten(1)
# LAYER_FNS = {'conv1': act_conv1_batch, 'conv2': act_conv2_batch, 'conv3': act_conv3_batch}
# LAYER_FNS = {'conv1': act_conv1_batch, 'conv2': act_conv2_batch}
LAYER_FNS = {'conv1': act_conv1_batch}

# ─────────── batched activation extractor ─────────────────────────────
def get_activation_for_ds_batch(model, imgs, act_fn, device='cuda', batch_size=64, num_workers=4):
    dataset = TensorDataset(imgs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    embs = []

    model.eval(); model.to(device)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        for i, (batch_imgs,) in enumerate(loader):
            batch = batch_imgs.to(device, non_blocking=True)
            out = act_fn(model, batch)
            embs.append(out.cpu())
    return torch.cat(embs, dim=0)

# ─────────── JIT-accelerated clustering ───────────────────────────────
@numba.njit(parallel=True)
def greedy_feature_clustering_jit(corr, gamma):
    N = corr.shape[0]
    lab = -np.ones(N, np.int32)
    vis = np.zeros(N, np.bool_)
    cid = 0
    for i in range(N):
        if vis[i]: continue
        stack = [i]; vis[i] = True; lab[i] = cid
        while stack:
            u = stack.pop()
            for v in range(N):
                if not vis[v] and abs(corr[u, v]) >= gamma:
                    vis[v] = True; lab[v] = cid; stack.append(v)
        cid += 1
    return lab, cid

# ─────────── batched interaction tensor with GPU corr ────────────────
def layer_interaction_tensor_batch(models, act_fn, imgs, device='cuda', k_pca=50, thresh=99.5, layer_key=None):
    proj_ls, comp_ls = [], []
    for m_idx, m in enumerate(tqdm(models, desc="IT models", leave=False)):
        A = get_activation_for_ds_batch(m, imgs, act_fn, device, batch_size=256, num_workers=4)
        A_np = A.cpu().numpy()
        pca = PCA(n_components=k_pca).fit(A_np)
        P = torch.from_numpy(pca.transform(A_np)).to(device)
        proj_ls.append(P.T)              # (k, N)
        comp_ls.append(torch.from_numpy(pca.components_).float())  # (k, D)

    X = torch.cat(proj_ls, dim=0)       # (M*k, N)
    X = X - X.mean(1, keepdim=True)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
    corr = torch.mm(X, X.t()).cpu().numpy()
    gamma_c = thresh
    gamma_d = np.percentile(np.abs(X.cpu().numpy()), thresh)
    labels, F = greedy_feature_clustering_jit(corr, gamma_c)
    M = len(models)
    N = X.shape[1]
    Omega = torch.zeros((M, N, F), dtype=torch.int8)
    X_np = X.cpu().numpy()
    for m_idx in range(M):
        for i in range(P.shape[1]):
            row = m_idx * P.shape[1] + i
            Omega[m_idx, np.abs(X_np[row]) >= gamma_d, labels[row]] = 1
    return Omega, comp_ls

# ─────────── GPA worker (batch inside) ───────────────────────────────
def _batch_grad(model, imgs, pvec, act_fn):
    activation = act_fn(model, imgs)
    score = (activation @ pvec).mean()

    grads = torch.autograd.grad(score, model.parameters(), retain_graph=False, allow_unused=True)
    return {n: (g.abs().mean().item() if g is not None else 0.0)
            for (n, _), g in zip(model.named_parameters(), grads)}

def _worker_gpa_batch(args):
    state_dict, root_state, imgs, comps, g_row, keep_row, act_fn, device_id = args
    torch.cuda.set_device(device_id)
    base = copy.deepcopy(root_state).to(device_id)
    model = copy.deepcopy(base); model.load_state_dict(state_dict)
    loader = DataLoader(TensorDataset(imgs), batch_size=512, shuffle=False, pin_memory=True)
    w_local = {n:0.0 for n,_ in model.named_parameters()}
    comps = comps.to(device_id)
    
    for t, (g_t, keep) in enumerate(zip(g_row, keep_row)):
        if not keep: continue
        pvec = comps[t % comps.size(0)]
        for batch_idx, (batch_imgs,) in enumerate(loader):
            batch = batch_imgs.to(device_id, non_blocking=True)
            grads = _batch_grad(model, batch, pvec, act_fn)
            for n,v in grads.items(): w_local[n] += g_t * v
    del model, loader, comps
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return w_local

# ─────────── parallel GPA with shared tensors ────────────────────────
def grad_importance_map_parallel_batch(models, root_model, device, imgs, comp_ls, Omega,
                                       act_fn, models_per_gpu=10, max_gpus=1, top_freq=0.0):
    M = Omega.shape[0]
    g = Omega.sum(1).float()
    g /= (g.max(1, keepdim=True).values + 1e-12)
    keep = g >= torch.quantile(g, 1-top_freq, dim=1, keepdim=True) if top_freq else torch.ones_like(g, dtype=torch.bool)
    tasks = []
    for m_idx in range(M):
        gpu_id = (m_idx // models_per_gpu) % max_gpus
        tasks.append((models[m_idx].state_dict(), root_model,
                      imgs, comp_ls[m_idx], g[m_idx].cpu(), keep[m_idx].cpu(),
                      act_fn, gpu_id))
    with ProcessPoolExecutor(max_workers=min(len(tasks), max_gpus*models_per_gpu)) as ex:
        results = list(tqdm(ex.map(_worker_gpa_batch, tasks), total=len(tasks), desc="GPA-par"))
    w_map = {n:0.0 for n,_ in models[0].named_parameters()}
    for res in results:
        for k,v in res.items(): w_map[k] += v
    tot = sum(w_map.values()) + 1e-12
    
    return {k: torch.as_tensor(v/tot, dtype=torch.float32, device=device) for k,v in w_map.items()}

# ─────────── weighted avg (unchanged, vectorized) ───────────────────
def weighted_avg_param(name, client_states, wmaps, eps=1e-12):
    stack = torch.stack([cs[name] for cs in client_states], dim=0).float()
    raw_w = torch.tensor([wm[name] for wm in wmaps], dtype=stack.dtype, device=stack.device)
    w_min, w_max = raw_w.min(), raw_w.max()
    norm_w = (raw_w - w_min) / (w_max - w_min + eps) if w_max> w_min else torch.ones_like(raw_w)
    norm_w = norm_w.view(-1, *[1]*(stack.ndim-1))
    return (stack * norm_w).mean(dim=0)

# ─────────── inverse weighted avg (중요도 낮을수록 가중치 높임) ──────────────
def inverse_weighted_avg_param(name, client_states, wmaps, eps=1e-12):
    stack = torch.stack([cs[name] for cs in client_states], dim=0).float()
    raw_w = torch.tensor([wm[name] for wm in wmaps], dtype=stack.dtype, device=stack.device)
    w_min, w_max = raw_w.min(), raw_w.max()
    # 중요도가 낮을수록 가중치가 높아지도록 반전
    inv_w = w_max - raw_w + eps
    norm_w = (inv_w - inv_w.min()) / (inv_w.max() - inv_w.min() + eps) if inv_w.max() > inv_w.min() else torch.ones_like(inv_w)
    norm_w = norm_w.view(-1, *[1]*(stack.ndim-1))
    return (stack * norm_w).mean(dim=0)

# ─────────── FedOpt 클래스 예시 통합 (레이어별 처리) ────────────────
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
        
        # imgs 공유
        imgs = []
        for _, img in self.merged_data:
            img_t = torch.from_numpy(img.transpose(2,0,1)).float() if isinstance(img, np.ndarray) else img.float()
            imgs.append(img_t)
        self.imgs = torch.stack(imgs).share_memory_()

    def aggregate(self, k_pca=50, models_per_gpu=10, max_gpus=1, top_freq=0.1):
        client_states = self.clientsModels
        client_models = []
        for state in client_states:
            m = copy.deepcopy(self.resultRootModel).to(self.device)
            m.load_state_dict(state)
            client_models.append(m)

        # 1) 레이어별 상호작용 텐서 생성 및 중요도 매핑
        layer_wmaps = {}
        for layer_key, act_fn in LAYER_FNS.items():
            Omega, comp_ls = layer_interaction_tensor_batch(client_models, act_fn, self.imgs, 
                device=self.device, k_pca=k_pca, layer_key=layer_key
            )
            wmap = grad_importance_map_parallel_batch(
                client_models, self.resultRootModel, self.device, self.imgs,
                comp_ls, Omega, act_fn,
                models_per_gpu, max_gpus, top_freq
            )
            layer_wmaps[layer_key] = [wmap]

        # 2) 레이어별 가중치 평균 적용
        new_state = {}
        for name in client_states[0].keys():
            # fc 레이어는 일반 FedAvg 적용
            if name.startswith("fc") or name.startswith("conv3") or name.startswith("conv1"):
                stack = torch.stack([cs[name] for cs in client_states], dim=0).float()
                new_state[name] = stack.mean(dim=0)
                continue
            for layer_key in LAYER_FNS:
                if name.startswith(layer_key):
                    # new_state[name] = weighted_avg_param(name, client_states, layer_wmaps[layer_key])
                    new_state[name] = weighted_avg_param(name, client_states, layer_wmaps[layer_key])
                    break
        
        torch.cuda.empty_cache()
        self.resultRootModel.load_state_dict(new_state, strict=False)
        return self.resultRootModel