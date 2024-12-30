import multiprocessing
import time
from multiprocessing import Process
import wandb
import json


class WandbClient(Process):
    def __init__(self, config):
        super().__init__()
        self.q = multiprocessing.Queue()
        self.config = config
        self._terminate = multiprocessing.Event()

    def register_client_metrics(self, wandb_run):
        basicConfig = self.config['basicInfo']
        clientConfig = self.config['clients']
        num_clients = int(basicConfig['numClient'])

        # Define basic metrics
        wandb_run.define_metric("custom_step")
        wandb_run.define_metric("time")

        wandb_run.define_metric("server/performance/server round time", step_metric="custom_step")
        wandb_run.define_metric("server/performance/server aggregated validation loss", step_metric="custom_step")
        wandb_run.define_metric("server/performance/server aggregated accuracy", step_metric="custom_step")

        for dataset_name in basicConfig['dataset']:
            wandb_run.define_metric(f"server/performance - {dataset_name}/server aggregated validation loss - {dataset_name}", step_metric="custom_step")
            wandb_run.define_metric(f"server/performance - {dataset_name}/server aggregated accuracy - {dataset_name}", step_metric="custom_step")

            for i in range(basicConfig['numClass']):
                wandb_run.define_metric(f"server/performance - {dataset_name}/aggregated class {i} accuracy - {dataset_name}",step_metric="custom_step")
                wandb_run.define_metric(f"server/performance - {dataset_name}/aggregated class {i} accuracy - {dataset_name}", step_metric="custom_step")

        # Define client-specific metrics
        for i in range(num_clients):
            wandb_run.define_metric(f"client/performance/train/loss/client{i} training loss", step_metric="custom_step")
            wandb_run.define_metric(f"client/performance/train/accuracy/client{i} training accuracy", step_metric="custom_step")
            wandb_run.define_metric(f"client/performance/validation/loss/client{i} validation loss", step_metric="custom_step")
            wandb_run.define_metric(f"client/performance/validation/accuracy/client{i} validation accuracy", step_metric="custom_step")
            wandb_run.define_metric(f"client/performance/trainTime/lastTrainTime/client{i} lastTrainTime", step_metric="custom_step")
            wandb_run.define_metric(f"client/performance/trainTime/avgTrainTime/client{i} avgTrainTime", step_metric="custom_step")
            wandb_run.define_metric(f"client/performance/pre-validation/loss/client{i} training loss", step_metric="custom_step")
            wandb_run.define_metric(f"client/performance/pre-validation/accuracy/client{i} training accuracy", step_metric="custom_step")
            wandb_run.define_metric(f"client/efficiency/waitingTime/client{i} waiting time", step_metric="custom_step")
            wandb_run.define_metric(f"client/metadata/learningRate-origin/client{i} origin lr", step_metric="custom_step")
            wandb_run.define_metric(f"client/metadata/learningRate-adjusted/client{i} adjusted lr", step_metric="custom_step")
            wandb_run.define_metric(f"client/metadata/penalty_reg/client{i} reg", step_metric="custom_step")
            wandb_run.define_metric(f"client/metadata/temperature/client{i} T", step_metric="custom_step")
            wandb_run.define_metric(f"client/metadata/epoch/client{i} epoch", step_metric="custom_step")
            wandb_run.define_metric(f"client/metadata/datasize/client{i} dataSize", step_metric="custom_step")
            wandb_run.define_metric(f"client/metadata/batchsize/client{i} batchSize", step_metric="custom_step")

        # Define client type-specific metrics
        wandb_run.define_metric(f"clientType/performance/validation/loss/client type all validation accuracy", step_metric="custom_step")
        wandb_run.define_metric(f"clientType/performance/pre-validation/loss/client type all validation accuracy", step_metric="custom_step")
        for type_num in range(len(clientConfig)):
            wandb_run.define_metric(f"clientType/performance/pre-validation/loss/client type{type_num} validation loss", step_metric="custom_step")
            wandb_run.define_metric(f"clientType/performance/pre-validation/accuracy/client type{type_num} validation accuracy", step_metric="custom_step")
            wandb_run.define_metric(f"clientType/performance/validation/loss/client type{type_num} validation loss", step_metric="custom_step")
            wandb_run.define_metric(f"clientType/performance/validation/accuracy/client type{type_num} validation accuracy", step_metric="custom_step")
            wandb_run.define_metric(f"clientType/efficiency/waitingTime/client type{type_num} avg waiting time", step_metric="custom_step")

    def run(self):
        # Initialize wandb inside the child process
        self.wandb_run = wandb.init(project=self.config['basicInfo']['projectName'],
                                    name=self.config['basicInfo']['testName'],
                                    tags=self.config['basicInfo']['tags'],
                                    group=self.config['basicInfo']['groupName'],
                                    config=self.config, reinit=True)
        self.register_client_metrics(self.wandb_run)
        print('Wandb client online')

        while not self._terminate.is_set():
            try:
                if not self.q.empty():
                    buffData = self.q.get()
                    if buffData == "TERMINATE":
                        self._terminate.set()
                        break
                    key, value, step = buffData
                    self.send_log(key=key, data=value, step=step)
                else:
                    time.sleep(0.01)  # Prevent busy waiting
            except Exception as e:
                print(f"Error in WandbClient run loop: {e}")
                break

        # Finish the wandb run gracefully
        wandb.finish()

    def send_log(self, key, data, step):
        try:
            log_dict = {
                key: data,
                "custom_step": step
            }
            self.wandb_run.log(log_dict, step=step)
        except Exception as e:
            print(f"Error while logging to wandb: {e}")

    def get_queue(self):
        return self.q

    def terminate_client(self):
        self.q.put("TERMINATE")
