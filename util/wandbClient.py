import multiprocessing
import time
from multiprocessing import Process
import wandb


class wandbClient(Process):
    def __init__(self, config):
        super().__init__()
        self.q = multiprocessing.Queue()
        basicConfig = config['basicInfo']
        clientConfig = config['clients']
        self.wandbClient = wandb.init(project=basicConfig['projectName'], config=config)
        self.wandbClient.define_metric("custom_step")
        self.wandbClient.define_metric("time")
        self.registerClientMetric("server/performance/server aggregated validation loss")
        self.registerClientMetric("server/performance/server aggregated accuracy")
        self.registerClientMetric("server/performance/server round time")
        print('wandb client online')

        for i in range(int(basicConfig['numClient'])):
            self.registerClientMetric(f"client/performance/train/loss/client{i} training loss")
            self.registerClientMetric(f"client/performance/train/accuracy/client{i} training accuracy")

            self.registerClientMetric(f"client/performance/validation/loss/client{i} validation loss")
            self.registerClientMetric(f"client/performance/validation/accuracy/client{i} validation accuracy")

            self.registerClientMetric(f"client/performance/trainTime/lastTrainTime/client{i} lastTrainTime")
            self.registerClientMetric(f"client/performance/trainTime/avgTrainTime/client{i} avgTrainTime")

            self.registerClientMetric(f"client/performance/pre-validation/loss/client{i} training loss")
            self.registerClientMetric(f"client/performance/pre-validation/accuracy/client{i} training accuracy")

            self.registerClientMetric(f"client/efficiency/waitingTime/client{i} waiting time")

            self.registerClientMetric(f"client/metadata/learningRate-origin/client{i} origin lr")
            self.registerClientMetric(f"client/metadata/learningRate-adjusted/client{i} adjusted lr")
            self.registerClientMetric(f"client/metadata/temperature/client{i} T")
            
            self.registerClientMetric(f"client/metadata/epoch/client{i} epoch")
            self.registerClientMetric(f"client/metadata/datasize/client{i} dataSize")
            self.registerClientMetric(f"client/metadata/batchsize/client{i} batchSize")

        for typeNum in range(len(clientConfig)):
            self.registerClientMetric(
                f"clientType/performance/pre-validation/loss/client type{typeNum} validation loss")
            self.registerClientMetric(
                f"clientType/performance/pre-validation/accuracy/client type{typeNum} validation accuracy")

            self.registerClientMetric(f"clientType/performance/validation/loss/client type{typeNum} validation loss")
            self.registerClientMetric(
                f"clientType/performance/validation/accuracy/client type{typeNum} validation accuracy")
            self.registerClientMetric(f"clientType/efficiency/waitingTime/client type{typeNum} avg waiting time")

    def registerClientMetric(self, key):
        self.wandbClient.define_metric(key, step_metric="custom_step")

    def getQueue(self):
        return self.q

    def run(self):
        while True:
            try:
                buffData = self.q.get()
                key = buffData[0]
                value = buffData[1]
                step = buffData[2]
                self.sendLog(key=key, data=value, step=step)
            finally:
                time.sleep(0.01)

    def sendLog(self, key, data, step):
        log_dict = {
            key: data,
            "custom_step": step
        }
        self.wandbClient.log(log_dict)
