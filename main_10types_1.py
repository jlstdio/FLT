import copy
import multiprocessing
import os
import time
from dataPrepare.data_prepare_manager import create_dataset_dict, select_dataset
from dataPrepare.noniid import *
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
    sharedConfig = config['client_server_shared']
    networkConfig = config['networkConfig']

    '''copy necessary information'''
    serverConfig['model'] = sharedConfig['model']
    serverConfig['costFunc'] = sharedConfig['costFunc']

    for single_client_config in clientConfig:
        single_client_config['model'] = sharedConfig['model']
        single_client_config['costFunc'] = sharedConfig['costFunc']

        if str(basicConfig['aggregate_mode']).__contains__("fisher"):
            single_client_config['update_fisher_every'] = serverConfig['update_cluster_every']
    ''''''

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
    clientDataset_cifar10_jg, serverTestDataset_cifar10_jg, classes_cifar10 = select_dataset(dataset_name=dataset_list[0],
                                                                                                         client_subset_start_point=0.0,
                                                                                                         client_subset_ratio=0.1,
                                                                                                         server_subset_ratio=1.0)

    clientDataset_cifar10_jo, serverTestDataset_cifar10_jo, _ = select_dataset(dataset_name=dataset_list[1],
                                                                                           client_subset_start_point=0.1,
                                                                                           client_subset_ratio=0.1,
                                                                                           server_subset_ratio=1.0)

    clientDataset_cifar10_jp, serverTestDataset_cifar10_jp, _ = select_dataset(dataset_name=dataset_list[2],
                                                                                         client_subset_start_point=0.2,
                                                                                         client_subset_ratio=0.1,
                                                                                         server_subset_ratio=1.0)

    clientDataset_cifar10_r1, serverTestDataset_cifar10_r1, _ = select_dataset(dataset_name=dataset_list[3],
                                                                                       client_subset_start_point=0.3,
                                                                                       client_subset_ratio=0.1,
                                                                                       server_subset_ratio=1.0)

    clientDataset_cifar10_r2, serverTestDataset_cifar10_r2, _ = select_dataset(dataset_name=dataset_list[4],
                                                                               client_subset_start_point=0.4,
                                                                               client_subset_ratio=0.1,
                                                                               server_subset_ratio=1.0)

    clientDataset_cifar10_r3, serverTestDataset_cifar10_r3, _ = select_dataset(dataset_name=dataset_list[5],
                                                                               client_subset_start_point=0.5,
                                                                               client_subset_ratio=0.1,
                                                                               server_subset_ratio=1.0)

    clientDataset_cifar10_r4, serverTestDataset_cifar10_r4, _ = select_dataset(dataset_name=dataset_list[6],
                                                                               client_subset_start_point=0.6,
                                                                               client_subset_ratio=0.1,
                                                                               server_subset_ratio=1.0)

    clientDataset_cifar10_lp, serverTestDataset_cifar10_lp, _ = select_dataset(dataset_name=dataset_list[7],
                                                                               client_subset_start_point=0.7,
                                                                               client_subset_ratio=0.1,
                                                                               server_subset_ratio=1.0)

    clientDataset_cifar10_bp, serverTestDataset_cifar10_bp, _ = select_dataset(dataset_name=dataset_list[8],
                                                                               client_subset_start_point=0.8,
                                                                               client_subset_ratio=0.1,
                                                                               server_subset_ratio=1.0)

    clientDataset_cifar10_bs, serverTestDataset_cifar10_bs, _ = select_dataset(dataset_name=dataset_list[9],
                                                                               client_subset_start_point=0.9,
                                                                               client_subset_ratio=0.1,
                                                                               server_subset_ratio=1.0)

    '''
    "dataset": ["cifar-10_jg", "cifar-10_jo", "cifar-10_jp",
                "cifar-10_r1", "cifar-10_r2", "cifar-10_r3", "cifar-10_r4",
                "cifar-10_lp", "cifar-10_bp", "cifar-10_bs"],
    '''

    serverTestDataset_list = [serverTestDataset_cifar10_jg,
                              serverTestDataset_cifar10_jo,
                              serverTestDataset_cifar10_jp,
                              serverTestDataset_cifar10_r1,
                              serverTestDataset_cifar10_r2,
                              serverTestDataset_cifar10_r3,
                              serverTestDataset_cifar10_r4,
                              serverTestDataset_cifar10_lp,
                              serverTestDataset_cifar10_bp,
                              serverTestDataset_cifar10_bs]

    classes = classes_cifar10

    clientsDatasetDict = create_dataset_dict(dataset_distribution_name=basicConfig['dataset_distribution'],
                                             clientDataset_list=[clientDataset_cifar10_jg,
                                                                 clientDataset_cifar10_jo,
                                                                 clientDataset_cifar10_jp,
                                                                 clientDataset_cifar10_r1,
                                                                 clientDataset_cifar10_r2,
                                                                 clientDataset_cifar10_r3,
                                                                 clientDataset_cifar10_r4,
                                                                 clientDataset_cifar10_lp,
                                                                 clientDataset_cifar10_bp,
                                                                 clientDataset_cifar10_bs],
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

    if sharedConfig['costFunc'] == 'BCEloss' and sharedConfig['model'] != 'testModel_w_softmax':
        print('Binary CE criterion needs softmax contained model')
        exit()

    if sharedConfig['model'] == 'testModel_w_softmax':
        from model.testModel_w_softmax import testNN_w_Softmax
        modelToLoad = testNN_w_Softmax(numClasses)
    elif sharedConfig['model'] == 'testModel_wo_softmax_3_layer':
        from model.testModel_wo_softmax_3_layer import testNN_wo_Softmax_3_layer
        modelToLoad = testNN_wo_Softmax_3_layer(numClasses)
    elif sharedConfig['model'] == 'testModel_wo_softmax_5_layer':
        from model.testModel_wo_softmax_5_layer import testNN_wo_Softmax_5_layer
        modelToLoad = testNN_wo_Softmax_5_layer(numClasses)
    elif sharedConfig['model'] == 'testModel_wo_softmax_7_layer':
        from model.testModel_wo_softmax_7_layer import testNN_wo_Softmax_7_layer
        modelToLoad = testNN_wo_Softmax_7_layer(numClasses)
    elif sharedConfig['model'] == 'testModel_wo_softmax_more_filters':
        from model.testModel_wo_softmax_more_filters import testNN_wo_Softmax_more_filters
        modelToLoad = testNN_wo_Softmax_more_filters(numClasses)
    elif sharedConfig['model'] == 'testModel_wo_softmax_bigger_filters':
        from model.testModel_wo_softmax_bigger_filters import testNN_wo_Softmax_bigger_filters
        modelToLoad = testNN_wo_Softmax_bigger_filters(numClasses)
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
    dltAllFiles(basicConfig['clientsNegotiationFolderPath'])
    dltAllFiles(basicConfig['receivedProfilePath'])
    dltAllFiles(basicConfig['memorizedPthPath'])
    dltAllFiles(basicConfig['aggregateFisherPath'])
    dltAllFiles(basicConfig['aggregateFilePath'])
    dltAllFiles(basicConfig['receivedFisherPath'])


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    networkConfigRoot = './config/networkConfig'
    dataConfigRoot = './config/datasetConfig'

    networkConfig_PathList = [f'{networkConfigRoot}/feature_space_guiding/config_fedavg_3_layer_cka_MD_RP_3.json',
                              f'{networkConfigRoot}/feature_space_guiding/config_fedavg_3_layer_cka_MD_RP_4.json',
                              f'{networkConfigRoot}/feature_space_guiding/config_fedavg_3_layer_cka_MD_RP_5.json']

    dataConfig_PathList = [f'{dataConfigRoot}/dataConfig_dirichlet_prelim_not_mixed_type.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet_prelim_not_mixed_type.json',
                           f'{dataConfigRoot}/dataConfig_dirichlet_prelim_not_mixed_type.json']

    for network_configPath, data_configPath in zip(networkConfig_PathList, dataConfig_PathList):
        print(f'running with {network_configPath} | {data_configPath}')
        runner(network_configPath, data_configPath)

    print('end of program')
