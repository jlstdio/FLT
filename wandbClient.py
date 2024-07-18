import multiprocessing
import time
from multiprocessing import Process
import wandb

class wandbClient(Process):
    def __init__(self, config):
        super().__init__()
        self.q = multiprocessing.Queue()
        basicConfig = config['basicInfo']
        self.wandbClient = wandb.init(project=basicConfig['projectName'],config=config)
        print('wandb client online')


    def getQueue(self):
        return self.q


    def run(self):
        while True:
            buffData = self.q.get()
            key = buffData[0]
            value = buffData[1]
            self.sendLog(key=key, data=value)
            time.sleep(0.01)

    def sendLog(self, key, data):
        self.wandbClient.log({key: data})