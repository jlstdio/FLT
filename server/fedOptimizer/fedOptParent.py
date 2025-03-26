import copy
import time
import torch
from numba.cuda import is_available
from util.util import remove_module_prefix


def summary(model):
    modelDict = dict(model)
    for key in modelDict.keys():
        shape = modelDict[key].shape
        print(f'key {key} | {shape}')


class fedOptParent:
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

    def flush(self):
        self.resultRootModel = copy.deepcopy(self.rootModelStatic)
        self.clientsModels = []
        self.clientsLosses = []

    def registerPth(self, path):
        if path is None:
            return

        model_state_dict = torch.load(path, map_location=self.device, weights_only=True) # torch.load(path, map_location=torch.device('cpu'))
        # model_state_dict = remove_module_prefix(model_state_dict) # 만약 모델이 서버와 다른 architecture 생성되었다면 실행
        rootModel = copy.deepcopy(self.rootModelStatic)

        model = rootModel.to(self.device)
        model.load_state_dict(model_state_dict)

        self.clientsModels.append(model.state_dict())
        file_name = path.split('/')[-1].split('.')[0]
        client_id_from_name = int(file_name.split('_')[0])
        
        if self.additionalInfo is not None:
            if 'cluster_info' in self.additionalInfo.keys():
                type_info_by_clients = self.additionalInfo['cluster_info']
                self.clients_types.append(int(type_info_by_clients[client_id_from_name]))

        self.clients_ids.append(int(client_id_from_name))

        # print(f'[TEST] client{client_id_from_name} of type {type_info_by_clients[client_id_from_name]}: ')

    def aggregate(self):
        pass

    def afterWork(self):
        pass
