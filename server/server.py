import copy
from multiprocessing import Process
import os
import time
import numpy as np
from numba.cuda import is_available
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
from examinModel import examinModel

class PTHFileHandler(FileSystemEventHandler):
    def __init__(self, server):
        self.server = server

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pth'):
            file_name = os.path.basename(event.src_path)
            self.server.process_new_file(file_name)


class Server(Process):
    def __init__(self, rootModel, cudaId, flModel, examinDataset, serverConfig, basicConfig, currentRound, flipboard, turnFlag, sessionId, wandb):
        super(Server, self).__init__()
        self.device = None
        self.wandb = wandb
        self.serverConfig = serverConfig
        self.basicConfig = basicConfig
        self.cudaId = cudaId
        self.targetRound = serverConfig['flRound']
        self.reservedRootModel = rootModel
        self.rootModel = None
        self.flModel = flModel
        self.turnFlag = turnFlag
        self.sessionId = sessionId
        self.currentRound = currentRound
        self.participants = basicConfig['numClient']
        self.internalIdWithClients = self.participants
        self.status = [True for i in range(self.participants)]
        self.flipboard = flipboard
        self.examinDataset = examinDataset
        self.clientsList = [i for i in range(self.participants)]

        # mkdir
        self.pth_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "receivedPth"))
        os.makedirs(self.pth_folder, exist_ok=True)

        # root model init
        self.rootModel = copy.deepcopy(self.reservedRootModel)
        torch.save(self.rootModel.state_dict(), './server/rootModel/rootModel.pth')
        del self.rootModel

        print("Server online")

    def process_new_file(self, file_name):
        try:
            client_id = int(file_name.split('_')[0])
            if not self.status[client_id]:
                self.status[client_id] = True
                print(f"Received file from client {client_id}")

        except ValueError:
            print(f"Invalid file name format: {file_name}")

        if all(self.status):
            print('waiting for last one the upload file completely')
            flag = True
            while flag:
                flag = False
                for i in range(len(self.flipboard)):
                    if self.flipboard[i] == 0:
                        flag = True

                time.sleep(1)

            self.run_FL()

    def run_FL(self):
        pth_files = [os.path.join(self.pth_folder, f) for f in os.listdir(self.pth_folder) if f.endswith('.pth')]
        print(f"Running round {self.currentRound.value} FL with {len(pth_files)} clients")
        self.device = torch.device(f"cuda" if is_available() else "cpu")

        self.flModel.flush()

        for filePath in pth_files:
            self.flModel.registerPth(filePath)

        self.rootModel = copy.deepcopy(self.flModel.aggregate())
        torch.save(self.rootModel.state_dict(), f'./server/aggregatedPth/root_round{self.currentRound.value}.pth')
        torch.save(self.rootModel.state_dict(), './server/rootModel/rootModel.pth')

        examinManager = examinModel(self.internalIdWithClients, self.cudaId, self.examinDataset, self.serverConfig['examinData_batchSize'], self.rootModel, './server/rootModel/rootModel.pth')
        examinManager.loadData()
        loss, acc = examinManager.examin()
        self.wandb.log({f"server aggregated validation loss": loss})
        self.wandb.log({f"server aggregated accuracy": acc })

        del self.rootModel

        # Reset status and increment round
        self.status = [True] * self.participants
        for i in range(self.participants):
            self.turnFlag[i] = 0
            self.flipboard[i] = 1

        # Delete all pth files
        for file in pth_files:
            os.remove(file)

        if self.targetRound > self.currentRound.value:
            self.pickClients()
            self.currentRound.value += 1

        print(f'round is now {self.currentRound.value}')

    def pickClients(self):
        updateClientsPerRound = self.basicConfig['updateClientsPerRound']
        pickedClients = np.random.choice(self.clientsList, updateClientsPerRound, replace=False)
        session_id = 0

        for i in pickedClients:
            self.sessionId[i] = session_id
            session_id += 1
            self.status[i] = False
            self.turnFlag[i] = 1  # mark the client which is picked
            self.flipboard[i] = 0 # mark as file not sent


    def informRunToClients(self):
        print('informing to clients')
        self.pickClients()
        self.currentRound.value = 1

    def run(self):
        event_handler = PTHFileHandler(self)
        observer = Observer()
        observer.schedule(event_handler, self.pth_folder, recursive=False)
        observer.start()

        try:
            while True:
                if self.targetRound < self.currentRound.value:
                    print(f'server round is over {self.currentRound.value}/{self.targetRound}')
                    self.currentRound.value = -1
                    print(f'server will terminate after 10 sec')
                    time.sleep(10)
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

'''
if __name__ == "__main__":
    server = Server()
    server_thread = Thread(target=server.run)
    server_thread.start()

    # 예시: 다른 스레드에서 register와 getRound 메서드 호출
    print(f"Registered client ID: {server.register()}")
    print(f"Current round: {server.getRound()}")
    # 예시: 다른 스레드에서 register와 getRound 메서드 호출
    print(f"Registered client ID: {server.register()}")
    print(f"Current round: {server.getRound()}")
    # 예시: 다른 스레드에서 register와 getRound 메서드 호출
    print(f"Registered client ID: {server.register()}")
    print(f"Current round: {server.getRound()}")

    time.sleep(2)
    makeFile(0)
    time.sleep(2)
    makeFile(1)
    time.sleep(2)
    makeFile(2)
    time.sleep(2)

    server_thread.join() 
'''