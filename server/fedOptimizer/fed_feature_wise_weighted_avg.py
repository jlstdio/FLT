from server.fedOptimizer.fedOptParent import fedOptParent
import os, copy, numpy as np, torch
from torch import nn
from sklearn.decomposition import PCA
from numba.cuda import is_available
from typing  import Dict, List, Callable, Tuple
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)        # GPU-safe
torch.multiprocessing.set_sharing_strategy("file_system")  # ❶ FD 폭증 방지

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────── activation funcs ───────────────────────────────────────
def act_conv1(m,x): return m.relu(m.pool(m.conv1(x))).flatten(1)
def act_conv2(m,x): return m.relu(m.pool(m.conv2(m.relu(m.pool(m.conv1(x)))))).flatten(1)
def act_conv3(m,x):
    a=m.relu(m.pool(m.conv1(x))); a=m.relu(m.pool(m.conv2(a))); return m.relu(m.conv3(a)).flatten(1)

LAYER_FNS:Dict[str,Callable]={'conv1':act_conv1,'conv2':act_conv2,'conv3':act_conv3}

# ─────────── activation extractor ───────────────────────────────────
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
    
    # print(f"get_activation_for_ds: {len(embs)} images, {embs[0].shape} activations")
    return np.vstack(embs)
# ─────────── clustering util ────────────────────────────────────────
def greedy_feature_clustering(corr: np.ndarray, gamma: float):
    N = corr.shape[0]
    lab = -np.ones(N, int); vis = np.zeros(N, bool); cid = 0
    for i in range(N):
        if vis[i]: continue
        stack = [i]; vis[i] = True; lab[i] = cid
        while stack:
            u = stack.pop()
            for v in np.where(np.abs(corr[u]) >= gamma)[0]:
                if not vis[v]:
                    vis[v] = True; lab[v] = cid; stack.append(v)
        cid += 1
    return lab, cid

# ─────────── Ω + comps 생성 (기존 + comps 반환) ─────────────────────
def layer_interaction_tensor(models, act_fn, style_datasets, device=device,
                             k_pca=50, thresh=99.5):
    M, k = len(models), k_pca
    proj_ls, comp_ls = [], []
    for m in tqdm(models, desc="IT models", leave=False):
        A = get_activation_for_ds(m, copy.deepcopy(style_datasets), act_fn, device)
        pca = PCA(n_components=k).fit(A)
        proj_ls.append(torch.from_numpy(pca.transform(A).T))   # (k,N)
        comp_ls.append(torch.from_numpy(pca.components_).float())  # (k,D)

    X = torch.cat(proj_ls, 0).numpy()               # (M·k, N)
    X -= X.mean(1, keepdims=True)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9

    gamma_c, gamma_d = thresh, np.percentile(np.abs(X), thresh)
    labels, F = greedy_feature_clustering(X @ X.T, gamma_c)

    Ω = np.zeros((M, X.shape[1], F), np.int8)
    for m_idx in range(M):
        for i in range(k):
            row = m_idx * k + i
            Ω[m_idx, np.abs(X[row]) >= gamma_d, labels[row]] = 1
    return torch.from_numpy(Ω), proj_ls, comp_ls

# ─────────── GPA (single batch) ─────────────────────────────────────
def _batch_grad(model,x,pvec,act_fn,device_id):
    x=x.to(device_id, dtype=torch.float32).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    score=(act_fn(model,x) @ pvec.to(device_id)).mean()
    grads=torch.autograd.grad(score,model.parameters(),retain_graph=False,allow_unused=True)
    return {n: (g.abs().mean().item() if g is not None else 0.0)
            for (n,_),g in zip(model.named_parameters(),grads)}

# ─────────── 워커 함수 (프로세스별) ─────────────────────────────────
def _worker_gpa(args):
    (state_dict, root_model_state, imgs, comps, g_row, keep_row,
     act_key, device_id) = args

    torch.cuda.set_device(device_id)
    root_model = copy.deepcopy(root_model_state)
    model = copy.deepcopy(root_model).to(device_id)
    model.load_state_dict(state_dict)
    act_fn = act_key # LAYER_FNS[act_key]

    loader = torch.utils.data.DataLoader(         # index DataLoader
        range(len(imgs)), batch_size=32, shuffle=False)

    w_local = {n: 0.0 for n, _ in model.named_parameters()}
    k = comps.size(0); comps = comps.to(device_id)

    for t, (g_t, keep) in enumerate(zip(g_row, keep_row)):
        if not keep: continue
        pvec = comps[t % k]
        for idx_batch in loader:
            xb = torch.stack([imgs[i] for i in idx_batch]).to(device_id)
            grads = _batch_grad(model, xb, pvec, act_fn, device_id)
            for n, v in grads.items():
                w_local[n] += g_t * v
    return w_local

# ─────────── 병렬 GPA map ─────────────────────────────────────────
def grad_importance_map_parallel(models, root_model, imgs, comp_ls, Ω,
                                 act_fn_name,
                                 models_per_gpu=1, max_gpus=4,
                                 top_freq=0.0):

    M, _, F = Ω.shape
    g = Ω.sum(1).float(); g /= g.max(1, keepdim=True).values + 1e-12
    keep = g >= torch.quantile(g, 1 - top_freq, dim=1, keepdim=True) if top_freq \
           else torch.ones_like(g, dtype=torch.bool)

    tasks = []
    for m_idx, _ in enumerate(models):
        gpu_id = (m_idx // models_per_gpu) % max_gpus
        tasks.append((models[m_idx].state_dict(),
                      root_model,
                      imgs,
                      comp_ls[m_idx].cpu(),
                      g[m_idx].cpu(),
                      keep[m_idx].cpu(),
                      act_fn_name,
                      gpu_id))

    with ProcessPoolExecutor(max_workers=min(len(tasks), max_gpus*models_per_gpu)) as ex:
        results = list(tqdm(ex.map(_worker_gpa, tasks), total=len(tasks), desc="GPA-par"))

    w_map={n:0.0 for n,_ in models[0].named_parameters()}
    for res in results:
        for k,v in res.items(): w_map[k]+=v
    tot=sum(w_map.values())+1e-12
    # return {k:torch.tensor(v/tot,dtype=torch.float32,device=device) for k,v in w_map.items()}
    return {k: torch.as_tensor(v/tot, dtype=torch.float32, device=device).clone().detach() for k,v in w_map.items()}

# ─────────── weighted average util ────────────────────────────────
def weighted_avg_param(name, client_states, wmaps):
    stack=torch.stack([cs[name] for cs in client_states],0).float()
    w=torch.tensor([wm[name] for wm in wmaps],dtype=stack.dtype,device=stack.device
                   ).view(-1,*[1]*(stack.ndim-1))
    return (w*stack).sum(0)/(w.sum()+1e-9)

# ─────────── FedOpt class ─────────────────────────────────────────
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
        self.layer_datasets = {l: copy.deepcopy(self.merged_data) for l in LAYER_FNS}

    def aggregate(self, k_pca=50, models_per_gpu=10, max_gpus=1, top_freq=0.1):
        # Convert dataset images to CHW float32 format
        imgs = [torch.from_numpy(img.transpose(2,0,1)).float()
                       if isinstance(img, np.ndarray) else img.float()
                       for _, img in copy.deepcopy(self.merged_data)]

        client_states = self.clientsModels
        client_models = []
        for state in client_states:
            m = copy.deepcopy(self.resultRootModel).to(device)
            m.load_state_dict(state)
            client_models.append(m)

        layer_wmaps={l:[] for l in LAYER_FNS}
        for l,act_fn in tqdm(LAYER_FNS.items(),desc="layers"):
            Ω,_,comp_ls=layer_interaction_tensor(client_models,act_fn,
                                                 self.layer_datasets[l],
                                                 k_pca=k_pca,thresh=99.5)
            
            wm = grad_importance_map_parallel(client_models,
                                              self.resultRootModel,
                                              imgs,
                                              comp_ls,Ω,
                                              act_fn,
                                              models_per_gpu,max_gpus,top_freq)
            layer_wmaps[l].append(wm)

        new_state={}
        for l in LAYER_FNS:
            for k in [p for p in client_states[0].keys() if p.startswith(l)]:
                new_state[k]=weighted_avg_param(k,client_states,layer_wmaps[l])

        self.resultRootModel.load_state_dict(new_state,strict=False)
        return self.resultRootModel
