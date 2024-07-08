from multiprocessing import Process
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
from examinModel import examinModel
import wandb

from util.util import dltAllFiles


class PTHFileHandler(FileSystemEventHandler):
    def __init__(self, server):
        self.server = server

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pth'):
            file_name = os.path.basename(event.src_path)
            self.server.process_new_file(file_name)


def makeFile(idx):
    f = open(f"receivedPth/{idx}_file.pth", 'w')
    f.close()


class Server(Process):
    def __init__(self, rootModel, cudaId, flModel, examinDataset, config, currentRound, flipboard, wandb):
        super(Server, self).__init__()
        self.wandb = wandb
        self.config = config
        self.cudaId = cudaId
        self.targetRound = config['flRound']
        self.rootModel = rootModel
        self.flModel = flModel
        self.currentRound = currentRound
        self.participants = config['clients']
        self.internalIdWithClients = self.participants
        self.flipboard = flipboard # [False for i in range(self.participants)]
        self.status = [False for i in range(self.participants)]
        self.examinDataset = examinDataset
        self.pth_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".", "receivedPth"))
        dltAllFiles('./server/aggregatedPth')
        dltAllFiles('./server/receivedPth')
        dltAllFiles('./server/rootModel')
        torch.save(self.rootModel.state_dict(), './server/rootModel/rootModel.pth')
        os.makedirs(self.pth_folder, exist_ok=True)  # 폴더가 없으면 생성
        print("Server online")

    def process_new_file(self, file_name):
        try:
            client_id = int(file_name.split('_')[0])
            if client_id < len(self.status) and not self.status[client_id]:
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

        self.rootModel = self.flModel.aggregate()
        torch.save(self.rootModel.state_dict(), f'./server/aggregatedPth/root_round{self.currentRound.value}.pth')
        torch.save(self.rootModel.state_dict(), './server/rootModel/rootModel.pth')

        examinManager = examinModel(self.internalIdWithClients, self.cudaId, self.examinDataset, self.config['examinData_batchSize'], self.rootModel, './server/rootModel/rootModel.pth')
        examinManager.loadData()
        loss, acc = examinManager.examin()
        self.wandb.log({f"server aggregated validation loss": loss})
        self.wandb.log({f"server aggregated accuracy": acc })

        # Reset status and increment round
        self.status = [False] * len(self.status)
        for i in range(len(self.status)):
            self.flipboard[i] = 0
        
        if self.targetRound > self.currentRound.value:
            self.currentRound.value += 1

        print(f'all registered client status set to {self.status}')
        print(f'round is now {self.currentRound.value}')

        # Delete all pth files
        for file in pth_files:
            os.remove(file)

    def informRunToClients(self):
        print('informing to clients')
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