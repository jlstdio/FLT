import torch
import syft as sy


class virtualClient:
    def __init__(self, numClients):
        hook = sy.TorchHook(torch)

        # 가상 클라이언트 생성
        workers = [sy.VirtualWorker(hook, id=f"worker{i}") for i in range(numClients)]
        print(len(workers))
