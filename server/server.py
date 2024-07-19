import copy
import random
from multiprocessing import Process
import os
import time
import numpy as np
from numba.cuda import is_available
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
from examinModel import examinModel
from util.util import dltAllFiles


class PTHFileHandler(FileSystemEventHandler):
    def __init__(self, server):
        self.server = server

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pth'):
            file_name = os.path.basename(event.src_path)
            self.server.process_new_file(file_name)


class Server(Process):
    def __init__(self, rootModel, cudaId, flModel, examinDataset, serverConfig, basicConfig, currentRound, flipboard, turnFlag, sessionId, pickedClientsList, wandbQueue):
        super(Server, self).__init__()
        self.wandbQueue = wandbQueue
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
        self.pickedClientsList = pickedClientsList
        self.clientsList = [i for i in range(self.participants)]
        self.lastAcc = 0.0

        self.seed = basicConfig['seed']
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

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

        self.flModel.flush()

        for filePath in pth_files:
            self.flModel.registerPth(filePath)

        ## aggregating model
        self.rootModel = copy.deepcopy(self.flModel.aggregate())

        ## saving model
        torch.save(self.rootModel.state_dict(), f'./server/aggregatedPth/root_round{self.currentRound.value}.pth')
        torch.save(self.rootModel.state_dict(), './server/rootModel/rootModel.pth')

        examinManager = examinModel(self.internalIdWithClients, self.cudaId, self.examinDataset, self.serverConfig['examinData_batchSize'], self.rootModel, './server/rootModel/rootModel.pth')
        examinManager.loadData()
        loss, acc = examinManager.examin()

        self.wandbQueue.put(["server aggregated validation loss", loss])
        self.wandbQueue.put(["server aggregated accuracy", acc])

        # FL anomaly check
        '''
        if -20.0 > acc - self.lastAcc:
            print(f'anomaly detected at {self.currentRound.value}')
            torch.save(self.rootModel.state_dict(), f'./util/errorModel/rootModel_errored_at{self.currentRound.value}.pth')
            self.currentRound = self.targetRound + 1
            print('ejecting')
        else:
            dltAllFiles('./util/lastWorkingModel')
            torch.save(self.rootModel.state_dict(),f'./util/lastWorkingModel/rootModel_at_{self.currentRound.value}.pth')
        '''

        dltAllFiles('./util/lastWorkingModel')
        torch.save(self.rootModel.state_dict(), f'./util/lastWorkingModel/rootModel_at_{self.currentRound.value}.pth')

        self.lastAcc = acc

        # Reset status and increment round
        self.status = [True] * self.participants
        for i in range(self.participants):
            self.turnFlag[i] = 0
            self.flipboard[i] = 1

        # Delete all received pth files
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

        for i in range(updateClientsPerRound):
            self.pickedClientsList[i] = pickedClients[i]

    def startFL(self):
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
