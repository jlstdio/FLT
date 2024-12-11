from util.fisher import compute_fisher
import copy
from server.fedOptimizer.fedOptParent import fedOptParent
from util.util import loadData


class fedWeighedAvg_fisher(fedOptParent):
    def __init__(self, rootModel, cudaId, additionalInfo):
        super().__init__(rootModel, cudaId, additionalInfo)

    def aggregate(self):

        costFunc = self.additionalInfo['costFunc']
        dataset = self.additionalInfo['dataset']
        numClass = self.additionalInfo['numClass']

        data_loader = loadData(dataset, costFunc, numClass)

        # List to store Fisher Information for each client
        fishers = []

        # Compute Fisher Information for each client
        for i, clientStateDict in enumerate(self.clientsModels):
            # Load client model
            clientModel = copy.deepcopy(self.rootModelStatic)
            clientModel.load_state_dict(clientStateDict)
            clientModel.to(self.device)

            # Compute Fisher Information
            fisherInfo = compute_fisher(clientModel, data_loader, costFunc, self.device)
            fishers.append(fisherInfo)

        # Initialize the updated weights dictionary
        updated_weights = {}
        model_keys = self.clientsModels[0].keys()

        # Aggregate each parameter individually using Fisher Information weights
        for key in model_keys:
            # Collect Fisher Information for this parameter from all clients
            parameter_fishers = [fishers[i][key] for i in range(len(self.clientsModels))]

            # Sum over elements to get scalar importance for each client for this parameter
            importances = [fisher_param.abs().sum().item() for fisher_param in parameter_fishers]

            # Handle the case where all importances are zero
            total_importance = sum(importances)
            if total_importance == 0:
                # Assign equal weights if total importance is zero
                weights = [1.0 / len(importances)] * len(importances)
            else:
                # Normalize importances to get weights
                weights = [imp / total_importance for imp in importances]

            # Perform weighted sum of parameters across clients
            weighted_param = sum(self.clientsModels[i][key] * weights[i] for i in range(len(self.clientsModels)))
            updated_weights[key] = weighted_param

        # Load the updated weights into the server model
        self.resultRootModel.load_state_dict(updated_weights)
        return self.resultRootModel
