import copy
import os


##########################################################
# SELECTING & INITIATING AGGREGATOR ######################
##########################################################


def server_type_loader(basicConfig, serverConfig, reservedRootModel, cudaId, currentRound, examinDataset):
    flModel = None
    if basicConfig['aggregate_mode'] == 'fedAvg':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(reservedRootModel, cudaId)

    elif basicConfig['aggregate_mode'] == 'fed_prox':
        from server.fedOptimizer.fedAvg import fedAvg
        flModel = fedAvg(reservedRootModel, cudaId)

    elif basicConfig['aggregate_mode'] == 'fedAvg_w_mem':
        from server.fedOptimizer.fedAvg_w_memorization import fedAvg_w_mem

        memorized_pth_path = basicConfig['memorizedPthPath']
        pth_folder = basicConfig['receivedPthPath']
        pth_files = [os.path.join(pth_folder, f) for f in os.listdir(pth_folder) if f.endswith('.pth')]

        additional_info_dict = {
            'memorized_pth_path': memorized_pth_path,
            'maximum_pth_to_mix': serverConfig['maximum_pth_to_mix'],
            'server_round_mem': serverConfig['server_round_mem'],
            'pth_files': pth_files,
            'curRound': currentRound.value
        }
        flModel = fedAvg_w_mem(reservedRootModel, cudaId, additional_info_dict)
    elif basicConfig['aggregate_mode'] == 'fed_fisher_client':
        from server.fedOptimizer.fedCurv_fisher_calc_client import fedCurv_fisher_calc_client
        flModel = fedCurv_fisher_calc_client(reservedRootModel, cudaId, None)

    elif basicConfig['aggregate_mode'] == 'fedWeightedAvg_fisher':
        from server.fedOptimizer.fedWeightedAvg_fisher import fedWeighedAvg_fisher

        additional_info_dict = {
            'costFunc': serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': basicConfig['numClass']
        }

        flModel = fedWeighedAvg_fisher(reservedRootModel, cudaId, additional_info_dict)
    elif basicConfig['aggregate_mode'] == 'fed_fisher_server':
        from server.fedOptimizer.fedCurv_fisher_calc_server import fedCurv_fisher_calc_server

        additional_info_dict = {
            'costFunc': serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': basicConfig['numClass']
        }
        flModel = fedCurv_fisher_calc_server(reservedRootModel, cudaId, additional_info_dict)
    elif basicConfig['aggregate_mode'] == 'calm_fisher':
        from server.fedOptimizer.calm_fisher import calm_fisher

        additional_info_dict = {
            'costFunc': serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': basicConfig['numClass'],
            'curRound': currentRound.value,
            'fisher_patient': serverConfig['fisher_patient']
        }
        flModel = calm_fisher(reservedRootModel, cudaId, additional_info_dict)
    elif basicConfig['aggregate_mode'] == 'selective_fisher':
        from server.fedOptimizer.selective_fisher import selective_fisher

        additional_info_dict = {
            'costFunc': serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': basicConfig['numClass'],
            'top_percent': serverConfig['fisher_select']
        }
        flModel = selective_fisher(reservedRootModel, cudaId, additional_info_dict)
    elif basicConfig['aggregate_mode'] == 'calm_selective_fisher':
        from server.fedOptimizer.calm_selective_fisher import calm_selective_fisher

        additional_info_dict = {
            'costFunc': serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': basicConfig['numClass'],
            'top_percent': serverConfig['fisher_select'],
            'curRound': currentRound.value,
            'fisher_patient': serverConfig['fisher_patient']
        }
        flModel = calm_selective_fisher(reservedRootModel, cudaId, additional_info_dict)
    elif basicConfig['aggregate_mode'] == 'pretrained_fedAvg':
        from server.fedOptimizer.fedCurv_fisher_calc_server import fedCurv_fisher_calc_server

        additional_info_dict = {
            'costFunc': serverConfig['costFunc'],
            'dataset': copy.deepcopy(examinDataset),
            'numClass': basicConfig['numClass']
        }
        flModel = fedCurv_fisher_calc_server(reservedRootModel, cudaId, additional_info_dict)

    return flModel
