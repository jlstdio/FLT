{

  "basicInfo" : {

    "projectName": "FLT",
    "dataset": "cifar-10",
    "numClient" : 100,
    "clientsPerCuda": 6,
    "updateClientsPerRound": 10,
    "seed": 1234,
    "startingCuda": 2,
    "participantsInfo": "0:0.3|1:0.3|2:0.4",
    "errorFilePath": Where pth files with abnormal result is detected is saved
    "aggregateFilePath": Where every round of aggregated Root Model is saved
    "receivedPthPath": Where clients sends pth file to server
    "receivedDataPath": Where server receives client's train result and performance for logging
    "receivedProfilePath": Where server receives client's profile to calculate the client hyperparameter this round  
    "rootModelFilePath": Where lastest Root Model is saved and client is downloading the model from here
    "clientsMetadataFolderPath": Where all the clients saves their meta data including hyperparameter
    "clientsNegotiationFolderPath": Where server send negotiated hyperparameter json file to clients
    "enable_flid": enabling FLID algorithm

  },
  "networkConfig" : {

    "rate" : {
      "PenaltyRate" : 0.7,
      "RewardRate" : 0.4
    },
    "epoch" : {
      "PenaltyValue" : -2,
      "RewardValue" : 4
    },
    "batchSize" : {
      "PenaltyValue" : -5,
      "RewardValue" : 5
    },
    "dataSize" : {
      "PenaltyValue" : -5,
      "RewardValue" : 5
    }

  },
  "clients": [

    {
      "type": "fastest edge",
      "gpuClk": 1.0,
      "vram": 300,
      "cpuClk": 1.0,
      "ram": 300,
      "disk": 500,
      "memFrac": 0.01,
      "dataSetFrac" : 0.6,
      "learningRate": 0.01,
      "lr_decay_step": 1,
      "lr_decay": 1.0,
      "momentum": 0.9,
      "epoch": 20,
      "batchSize": 30,
      "iteration": 1000,
      "delayMin": 0,
      "delayMax": 0.1,
      "dataSize" : 1.0,
      "costFunc": "BCE loss",
      "optim" : "SGD",
      "reg": "L2",
      "model": "testNN"
    },
    {
      "type": "middle spec edge",
      "gpuClk": 1.0,
      "vram": 300,
      "cpuClk": 1.0,
      "ram": 300,
      "disk": 500,
      "memFrac": 0.01,
      "dataSetFrac" : 0.6,
      "learningRate": 0.01,
      "lr_decay_step": 1,
      "lr_decay": 1.0,
      "momentum": 0.9,
      "epoch": 20,
      "batchSize": 30,
      "iteration": 1000,
      "delayMin": 0.1,
      "delayMax": 0.2,
      "dataSize" : 1.0,
      "costFunc": "BCE loss",
      "optim" : "SGD",
      "reg": "L2",
      "model": "testNN"
    },
    {
      "type": "low spec edge",
      "gpuClk": 1.0,
      "vram": 300,
      "cpuClk": 1.0,
      "ram": 300,
      "disk": 500,
      "memFrac": 0.01,
      "dataSetFrac" : 0.6,
      "learningRate": 0.01,
      "lr_decay_step": 1,
      "lr_decay": 1.0,
      "momentum": 0.9,
      "epoch": 20,
      "batchSize": 30,
      "iteration": 1000,
      "delayMin": 1,
      "delayMax": 3,
      "dataSize" : 1.0,
      "costFunc": "BCE loss",
      "optim" : "SGD",
      "reg": "L2",
      "model": "testNN"
    }

  ],
  "server": {

    "flRound": 3000,
    "examinData_batchSize": 50

  }
}