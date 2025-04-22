import copy
import os


##########################################################
# SELECTING & INITIATING AGGREGATOR ######################
##########################################################


def server_type_loader(self, examinDataset):
    flModel = None
    if self.basicConfig['aggregate_mode'] == 'fedAvg' or self.basicConfig['aggregate_mode'] == 'fed_avg':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(self.reservedRootModel, self.cudaId)

    elif self.basicConfig['aggregate_mode'] == 'fed_prox' or self.basicConfig['aggregate_mode'] == 'partial_fed_prox':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(self.reservedRootModel, self.cudaId)

    elif self.basicConfig['aggregate_mode'] == 'fed_cka' or self.basicConfig['aggregate_mode'] == 'cka':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(self.reservedRootModel, self.cudaId)

    elif self.basicConfig['aggregate_mode'] == 'fed_cosine' or self.basicConfig['aggregate_mode'] == 'cosine':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(self.reservedRootModel, self.cudaId)

    elif self.basicConfig['aggregate_mode'] == 'fed_pearson' or self.basicConfig['aggregate_mode'] == 'pearson':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(self.reservedRootModel, self.cudaId)

    elif self.basicConfig['aggregate_mode'] == 'fed_l2' or self.basicConfig['aggregate_mode'] == 'l2':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(self.reservedRootModel, self.cudaId)

    elif self.basicConfig['aggregate_mode'] == 'fed_l_inf' or self.basicConfig['aggregate_mode'] == 'l_inf':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(self.reservedRootModel, self.cudaId)

    elif self.basicConfig['aggregate_mode'] == 'fed_em' or self.basicConfig['aggregate_mode'] == 'em':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(self.reservedRootModel, self.cudaId)

    elif self.basicConfig['aggregate_mode'] == 'fedAvg_w_mem':
        from server.fedOptimizer.fedAvg_w_memorization import fedAvg_w_mem

        memorized_pth_path = self.basicConfig['memorizedPthPath']
        pth_folder = self.basicConfig['receivedPthPath']
        pth_files = [os.path.join(pth_folder, f) for f in os.listdir(pth_folder) if f.endswith('.pth')]

        additional_info_dict = {
            'memorized_pth_path': memorized_pth_path,
            'maximum_pth_to_mix': self.serverConfig['maximum_pth_to_mix'],
            'server_round_mem': self.serverConfig['server_round_mem'],
            'pth_files': pth_files,
            'curRound': self.currentRound.value
        }
        flModel = fedAvg_w_mem(self.reservedRootModel, self.cudaId, additional_info_dict)
    elif self.basicConfig['aggregate_mode'] == 'fisher_client':
        from server.fedOptimizer.fedCurv_fisher_calc_client import fedCurv_fisher_calc_client

        additional_info_dict = {
            'curRound': self.currentRound.value,
            'update_fisher_every': self.serverConfig['update_cluster_every']
        }

        flModel = fedCurv_fisher_calc_client(self.reservedRootModel, self.cudaId, additional_info_dict)

    elif self.basicConfig['aggregate_mode'] == 'weighted_fed_avg_param_diff':
        from server.fedOptimizer.weighed_fed_avg_param_diff import weighed_fed_avg_param_diff

        additional_info_dict = {
            'costFunc': self.serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': self.basicConfig['numClass']
        }

        flModel = weighed_fed_avg_param_diff(self.reservedRootModel, self.cudaId, additional_info_dict)
    elif self.basicConfig['aggregate_mode'] == 'weighted_fed_avg_fisher':
        from server.fedOptimizer.weighed_fed_avg_fisher import weighed_fed_avg_fisher

        additional_info_dict = {
            'costFunc': self.serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': self.basicConfig['numClass']
        }

        flModel = weighed_fed_avg_fisher(self.reservedRootModel, self.cudaId, additional_info_dict)
    elif self.basicConfig['aggregate_mode'] == 'fisher_server':
        from server.fedOptimizer.fedCurv_fisher_calc_server import fedCurv_fisher_calc_server

        additional_info_dict = {
            'costFunc': self.serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': self.basicConfig['numClass'],
            'curRound': self.currentRound.value,
            'update_fisher_every': self.serverConfig['update_cluster_every']
        }
        flModel = fedCurv_fisher_calc_server(self.reservedRootModel, self.cudaId, additional_info_dict)
    elif self.basicConfig['aggregate_mode'] == 'calm_fisher':
        from server.fedOptimizer.calm_fisher import calm_fisher

        additional_info_dict = {
            'costFunc': self.serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': self.basicConfig['numClass'],
            'curRound': self.currentRound.value,
            'fisher_patient': self.serverConfig['fisher_patient']
        }
        flModel = calm_fisher(self.reservedRootModel, self.cudaId, additional_info_dict)
    elif self.basicConfig['aggregate_mode'] == 'selective_fisher':
        from server.fedOptimizer.selective_fisher import selective_fisher

        additional_info_dict = {
            'costFunc': self.serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': self.basicConfig['numClass'],
            'top_percent': self.serverConfig['fisher_select']
        }
        flModel = selective_fisher(self.reservedRootModel, self.cudaId, additional_info_dict)
    elif self.basicConfig['aggregate_mode'] == 'calm_selective_fisher':
        from server.fedOptimizer.calm_selective_fisher import calm_selective_fisher

        additional_info_dict = {
            'costFunc': self.serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': self.basicConfig['numClass'],
            'top_percent': self.serverConfig['fisher_select'],
            'curRound': self.currentRound.value,
            'fisher_patient': self.serverConfig['fisher_patient']
        }
        flModel = calm_selective_fisher(self.reservedRootModel, self.cudaId, additional_info_dict)
    elif self.basicConfig['aggregate_mode'] == 'pretrained_fedAvg':
        from server.fedOptimizer.fedCurv_fisher_calc_server import fedCurv_fisher_calc_server

        additional_info_dict = {
            'costFunc': self.serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': self.basicConfig['numClass']
        }
        flModel = fedCurv_fisher_calc_server(self.reservedRootModel, self.cudaId, additional_info_dict)

    elif self.basicConfig['aggregate_mode'] == 'fed_2way_avg':
        from server.fedOptimizer.fed_2way_avg import fed_2way_avg

        additional_info_dict = {
            'cluster_info': self.type_info_by_clients,
            'picked_clients': self.pickedClients,
            'rootModelFilePath': self.basicConfig['rootModelFilePath'],
            'latest_sub_roots_path': self.basicConfig['latest_sub_roots_path'],
            'resultPath': self.resultPath,
            'testName': self.basicConfig['testName'],
            'global_mix_ratio': self.serverConfig['global_mix_ratio'],
            'curRound': self.currentRound.value
            }

        flModel = fed_2way_avg(self.reservedRootModel, self.cudaId, additional_info_dict)
    
    elif self.basicConfig['aggregate_mode'] == 'fed_2way_distillation':
        from server.fedOptimizer.fed_2way_avg import fed_2way_avg

        additional_info_dict = {
            'cluster_info': self.type_info_by_clients,
            'picked_clients': self.pickedClients,
            'rootModelFilePath': self.basicConfig['rootModelFilePath'],
            'latest_sub_roots_path': self.basicConfig['latest_sub_roots_path'],
            'testName': self.basicConfig['testName'],
            'global_mix_ratio': self.serverConfig['global_mix_ratio'],
            }

        flModel = fed_2way_avg(self.reservedRootModel, self.cudaId, additional_info_dict)
    
    elif self.basicConfig['aggregate_mode'] == 'fed_2way_ewc':
        from server.fedOptimizer.fed_2way_avg import fed_2way_avg

        additional_info_dict = {
            'cluster_info': self.type_info_by_clients,
            'picked_clients': self.pickedClients,
            'rootModelFilePath': self.basicConfig['rootModelFilePath'],
            'testName': self.basicConfig['testName'],
            'global_mix_ratio': self.serverConfig['global_mix_ratio'],
            }

        flModel = fed_2way_avg(self.reservedRootModel, self.cudaId, additional_info_dict)

    return flModel
