import copy
import multiprocessing
import os
import time
from dataPrepare.data_prepare_manager import create_dataset_dict, select_dataset
from dataPrepare.noniid import *
from model.testModel_wo_softmax import testNN_wo_Softmax
from model.testModel_w_softmax import testNN_w_Softmax
from network.FLNetwork import FLNetwork
from server.server import Server
import json
import torch
from util.util import showDistribution, dltAllFiles
from util.wandbClient import WandbClient


def runner(networkConfigPath, dataConfigPath):
    with open(networkConfigPath, 'r') as file:
        config = json.load(file)

    clientConfig = config['clients']
    serverConfig = config['server']
    basicConfig = config['basicInfo']
    networkConfig = config['networkConfig']
    numClients = basicConfig['numClient']

    seed = basicConfig['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    resultRootPath = basicConfig["resultRoot"] + "/" + basicConfig['testName'] + "-" + str(round(time.time()))
    serverScoreFolderPath = resultRootPath + "/" + basicConfig["serverScoreFolderRoot"]
    clientScoreFolderPath = resultRootPath + "/" + basicConfig["clientScoreFolderRoot"]
    dataset_created_log_path = serverScoreFolderPath + basicConfig["dataset_created_log_path"]
    os.makedirs(serverScoreFolderPath, exist_ok=True)
    os.makedirs(clientScoreFolderPath, exist_ok=True)

    cleanUp_everything(basicConfig=basicConfig)

    testName = basicConfig['testName']

    ########################################
    # DATASET TYPE & DISTRIBUTION SETTINGS #
    ########################################

    clients_id_list = [i for i in range(numClients)]
    mnist_clients_ratio = 0.5

    dataset_list = basicConfig['dataset']
    clientDataset_cifar10_original, serverTestDataset_cifar10_original, classes_cifar10 = select_dataset(dataset_name=dataset_list[0],
                                                                                                         client_subset_start_point=0.0,
                                                                                                         client_subset_ratio=0.25,
                                                                                                         server_subset_ratio=1.0)

    clientDataset_cifar10_jittered, serverTestDataset_cifar10_jittered, _ = select_dataset(dataset_name=dataset_list[1],
                                                                                           client_subset_start_point=0.25,
                                                                                           client_subset_ratio=0.25,
                                                                                           server_subset_ratio=1.0)

    clientDataset_cifar10_rotated, serverTestDataset_cifar10_rotated, _ = select_dataset(dataset_name=dataset_list[2],
                                                                                         client_subset_start_point=0.5,
                                                                                         client_subset_ratio=0.25,
                                                                                         server_subset_ratio=1.0)

    clientDataset_cifar10_noised, serverTestDataset_cifar10_noised, _ = select_dataset(dataset_name=dataset_list[3],
                                                                                       client_subset_start_point=0.75,
                                                                                       client_subset_ratio=0.25,
                                                                                       server_subset_ratio=1.0)

    serverTestDataset_list = [serverTestDataset_cifar10_original,
                              serverTestDataset_cifar10_jittered,
                              serverTestDataset_cifar10_rotated,
                              serverTestDataset_cifar10_noised]

    classes = classes_cifar10

    clientsDatasetDict = create_dataset_dict(dataset_distribution_name=basicConfig['dataset_distribution'],
                                             clientDataset_list=[clientDataset_cifar10_original,
                                                                 clientDataset_cifar10_jittered,
                                                                 clientDataset_cifar10_rotated,
                                                                 clientDataset_cifar10_noised],
                                             classes=classes,
                                             batchSize=0,
                                             clients_id_list=clients_id_list,
                                             dataConfigPath=dataConfigPath,
                                             dataset_created_log_path=dataset_created_log_path,
                                             seed=seed)

    # check dataset
    print('[TEST] combined dict length: ' + str(len(clientsDatasetDict)))

    distributionSavePath = f'{resultRootPath}/clientsDataset {testName} - {int(round(time.time()))}'
    totalDistributionSet = showDistribution(clientsDatasetDict, classes, distributionSavePath)
    # ##################################################################

    ######################
    # CRITERION SETTINGS #
    ######################
    numClasses = basicConfig['numClass']
    modelToLoad = None
    if serverConfig['costFunc'] == 'CEloss':
        modelToLoad = testNN_wo_Softmax(numClasses)
    elif serverConfig['costFunc'] == 'BCEloss':
        modelToLoad = testNN_w_Softmax(numClasses)
    elif serverConfig['costFunc'] == 'BCEWithLogitsLoss':
        modelToLoad = testNN_wo_Softmax(numClasses)
    # ##################################################################

    wandbClientServer = WandbClient(config=config)
    wandbClientServer.start()
    wandbQueue = wandbClientServer.get_queue()

    modelToClients = copy.deepcopy(modelToLoad)
    network = FLNetwork(basicConfig=basicConfig,
                        clientsDatasetDict=clientsDatasetDict,
                        clientConfig=clientConfig,
                        networkConfig=networkConfig,
                        modelToLoad=modelToClients,
                        scorePath=clientScoreFolderPath,
                        wandbQueue=wandbQueue)
    network.start()

    serverRound, flipboard, turnFlag, sessionId, pickedClients = network.getSharedInfo()

    modelToServer = copy.deepcopy(modelToLoad)
    server = Server(rootModel=modelToServer,
                    examinDataset_list=serverTestDataset_list,
                    serverConfig=serverConfig,
                    basicConfig=basicConfig,
                    currentRound=serverRound,
                    flipboard=flipboard,
                    turnFlag=turnFlag,
                    sessionId=sessionId,
                    pickedClientsList=pickedClients,
                    resultPath=resultRootPath,
                    wandbQueue=wandbQueue,
                    totalDistributionSet=totalDistributionSet)

    server.start()

    server.join()
    network.join()
    wandbClientServer.terminate_client()
    wandbClientServer.join()
    torch.cuda.empty_cache()

    print('end of runner')


def cleanUp_everything(basicConfig):
    dltAllFiles(basicConfig['errorFilePath'])
    dltAllFiles(basicConfig['receivedPthPath'])
    dltAllFiles(basicConfig['receivedDataPath'])
    dltAllFiles(basicConfig['rootModelFilePath'])
    dltAllFiles(basicConfig['clientsMetadataFolderPath'])
    dltAllFiles(basicConfig['receivedProfilePath'])
    dltAllFiles(basicConfig['memorizedPthPath'])


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    networkConfigRoot = './config/networkConfig'
    dataConfigRoot = './config/datasetConfig'

    '''
    network_configPath, data_configPath = networkConfigRoot + '/' + sys.argv[1], dataConfigRoot + '/' + sys.argv[2]

    print(f'running with {network_configPath} | {data_configPath}')
    runner(network_configPath, data_configPath)
    '''

    networkConfig_PathList = [f'{networkConfigRoot}/domain_shift_performance/config_fisher_server_MD_mixed_RP_1.json',
                              f'{networkConfigRoot}/domain_shift_performance/config_fisher_server_MD_mixed_RP_2.json',
                              f'{networkConfigRoot}/domain_shift_performance/config_fedavg_MD_not_mixed_RP.json',
                              f'{networkConfigRoot}/domain_shift_performance/config_fisher_server_MD_not_mixed_RP_0.json',
                              f'{networkConfigRoot}/domain_shift_performance/config_fisher_server_MD_not_mixed_RP_1.json',
                              f'{networkConfigRoot}/domain_shift_performance/config_fisher_server_MD_not_mixed_RP_2.json']

    dataConfig_PathList = [f'{dataConfigRoot}/dataConfig_dirichlet_mixed_type.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet_mixed_type.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet_mixed_type.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet_mixed_type.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet_not_mixed_type.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet_not_mixed_type.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet_not_mixed_type.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet_not_mixed_type.json']

    for network_configPath, data_configPath in zip(networkConfig_PathList, dataConfig_PathList):
        print(f'running with {network_configPath} | {data_configPath}')
        runner(network_configPath, data_configPath)

    print('end of program')
