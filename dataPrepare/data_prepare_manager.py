

def select_dataset(dataset_name, client_subset_start_point=0.0, client_subset_ratio=1.0, server_subset_start_point=0.0, server_subset_ratio=1.0):
    dataloader = None
    (x_train, y_train), (x_test, y_test) = (None, None), (None, None)

    if dataset_name == 'cifar-10':
        from dataset.cifar10.cifar10DataLoader import cifar10Dataloader
        dataloader = cifar10Dataloader('./dataset/cifar10')
        (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    elif dataset_name == 'cifar-10_jitter':
        from dataset.cifar10_expanded.cifar10_expanded_dataloader import cifar10_expanded_dataloader
        dataloader = cifar10_expanded_dataloader('./dataset/cifar10_expanded')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_jitter_data()
    elif dataset_name == 'cifar-10_rotate':
        from dataset.cifar10_expanded.cifar10_expanded_dataloader import cifar10_expanded_dataloader
        dataloader = cifar10_expanded_dataloader('./dataset/cifar10_expanded')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_rotate_data()
    elif dataset_name == 'cifar-10_noise':
        from dataset.cifar10_expanded.cifar10_expanded_dataloader import cifar10_expanded_dataloader
        dataloader = cifar10_expanded_dataloader('./dataset/cifar10_expanded')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_noise_data()
    elif dataset_name == 'cifar-100':
        from dataset.cifar100.cifar100DataLoader import cifar100Dataloader
        dataloader = cifar100Dataloader('./dataset/cifar100')
        (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    elif dataset_name == 'mnist':
        from dataset.mnist.mnistDataLoader import mnistDataloader
        dataloader = mnistDataloader(data_dir='./dataset/mnist/')
        (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    elif dataset_name == 'svhn':
        from dataset.svhn.svhn_dataloader import svhnDataloader
        dataloader = svhnDataloader(data_dir='./dataset/svhn/')
        (x_train, y_train), (x_test, y_test) = dataloader.load_data()
    elif dataset_name == 'cifar-10_jr':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_jitter_data_red()
    elif dataset_name == 'cifar-10_jg':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_jitter_data_green()
    elif dataset_name == 'cifar-10_jo':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_jitter_data_orange()
    elif dataset_name == 'cifar-10_jp':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_jitter_data_purple()
    elif dataset_name == 'cifar-10_r1':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_rotate_data_1()
    elif dataset_name == 'cifar-10_r2':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_rotate_data_2()
    elif dataset_name == 'cifar-10_r3':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_rotate_data_3()
    elif dataset_name == 'cifar-10_r4':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_rotate_data_4()
    elif dataset_name == 'cifar-10_lp':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_freq_lowpass()
    elif dataset_name == 'cifar-10_bp':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_freq_bandpass()
    elif dataset_name == 'cifar-10_bs':
        from dataset.cifar10_expanded_3.cifar10_expanded_3_dataloader import cifar10_expanded_3_dataloader
        dataloader = cifar10_expanded_3_dataloader('./dataset/cifar10_expanded_3')
        _, _ = dataloader.load_data()
        (x_train, y_train), (x_test, y_test) = dataloader.get_freq_bandstop()

    classes = list(set(y_test))

    clientDataset_start_idx = int(len(y_train) * client_subset_start_point)
    clientDatasetSize = int(len(y_train) * client_subset_ratio)
    clientDataset = zip(y_train[clientDataset_start_idx:clientDataset_start_idx+clientDatasetSize],
                        x_train[clientDataset_start_idx:clientDataset_start_idx+clientDatasetSize])

    serverDataset_start_idx = int(len(y_train) * server_subset_start_point)
    serverDatasetSize = int(len(y_test) * server_subset_ratio)
    serverTestDataset = zip(y_test[serverDataset_start_idx:serverDataset_start_idx+serverDatasetSize],
                            x_test[serverDataset_start_idx:serverDataset_start_idx+serverDatasetSize])

    return clientDataset, serverTestDataset, classes


def create_dataset_dict(dataset_distribution_name, clientDataset_list, classes, clients_id_list, dataConfigPath, dataset_created_log_path, seed):
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

        # iidSplit(dataset_list, total_clients_id_list, dataset_created_log_path, seed=1234):
        from dataPrepare.iid import iidSplit
        clientsDatasetDict = iidSplit(dataset_list=clientDataset_list,
                                      classes=classes,
                                      total_clients_id_list=clients_id_list,
                                      configPath=dataConfigPath,
                                      dataset_created_log_path=dataset_created_log_path,
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

    elif dataset_distribution_name == 'dirichlet_vanilla_w_lossy_compress':
        from dataPrepare.noniid_w_compression import dirichletSplit_lossy_compress
        # (dataset_list, classes, total_clients_id_list, type_info, type_ratio, configPath, seed=1234):
        clientsDatasetDict = dirichletSplit_lossy_compress(dataset_list=clientDataset_list,
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
