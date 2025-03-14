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
        # self.clients_losses.append(client_loss)

    def aggregate(self):
        pass

    def afterWork(self):
        pass
