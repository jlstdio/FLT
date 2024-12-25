
def select_dataset(dataset_name, client_subset_ratio, server_subset_ratio=0.0):
    dataloader = None

    if dataset_name == 'cifar-10':
        from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
        dataloader = cifar10Dataloader('./dataset/cifar10')
    elif dataset_name == 'cifar-100':
        from dataset.cifar100.cifar100DataLoader import cifar100Dataloader
        dataloader = cifar100Dataloader('./dataset/cifar100')
    elif dataset_name == 'mnist':
        from dataset.mnist.mnistDataLoader import mnistDataloader
        dataloader = mnistDataloader(data_dir='./dataset/mnist/')
    elif dataset_name == 'svhn':
        from dataset.svhn.svhn_dataloader import svhnDataloader
        dataloader = svhnDataloader(data_dir='./dataset/svhn/')

    (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    classes = list(set(y_test))

    clientDatasetSize = int(len(y_train) * client_subset_ratio)
    clientDataset = zip(y_train[:clientDatasetSize], x_train[:clientDatasetSize])

    serverDatasetSize = int(len(y_test) * server_subset_ratio)
    serverTestDataset = zip(y_test[:serverDatasetSize], x_test[:serverDatasetSize])

    return clientDataset, serverTestDataset, classes


def create_dataset_dict(dataset_distribution_name, clientDataset_list, classes, batchSize, clients_id_list, dataConfigPath, dataset_created_log_path, seed):
    clientsDatasetDict = None
    if dataset_distribution_name == 'iid':
        # TODO: iidSplit 코드 수정하여야함
        '''
        변겅
            dataset -> dataset_list
        추가
            type_info
            type_ratio            
        '''
        from dataPrepare.iid import iidSplit
        clientsDatasetDict = iidSplit(dataset=clientDataset_list,
                                      classes=classes,
                                      batchSize=batchSize,
                                      clients_id_list=clients_id_list,
                                      seed=seed)

    elif dataset_distribution_name == 'dirichlet_vanilla':
        from dataPrepare.noniid import dirichletSplit
        # (dataset_list, classes, total_clients_id_list, type_info, type_ratio, configPath, seed=1234):
        clientsDatasetDict = dirichletSplit(dataset_list=clientDataset_list,
                                            classes=classes,
                                            total_clients_id_list=clients_id_list,
                                            configPath=dataConfigPath,
                                            dataset_created_log_path=dataset_created_log_path,
                                            seed=seed)

    elif dataset_distribution_name == 'dirichlet_strict_equal':
        # TODO: dirichlet_strict_equal 코드 수정하여야함
        '''
        변겅
            dataset -> dataset_list
        추가
            type_info
            type_ratio            
        '''
        from dataPrepare.noniid import dirichlet_equal_split
        clientsDatasetDict = dirichlet_equal_split(dataset=clientDataset_list,
                                                   classes=classes,
                                                   alpha=1.0,
                                                   clients_id_list=clients_id_list,
                                                   seed=seed)

    elif dataset_distribution_name == 'pathological':
        # TODO: pathological 코드 수정하여야함
        '''
        변겅
            dataset -> dataset_list
        추가
            type_info
            type_ratio            
        '''
        from dataPrepare.noniid import pathologicalSplit
        clientsDatasetDict = pathologicalSplit(dataset=clientDataset_list,
                                               classes=classes,
                                               clients_id_list=clients_id_list,
                                               configPath=dataConfigPath,
                                               seed=seed)

    '''
    elif dataset_distribution_name == 'dirichlet_diff_by_type':
        from dataPrepare.partiallyNonIid import difference_bias_by_type
        clientsDatasetDict = difference_bias_by_type(clientDataset,
                                                     classes,
                                                     configPath=dataConfigPath,
                                                     seed=seed)
    '''

    return clientsDatasetDict
