import numpy as np


def iidSplit(dataset, classes, batchSize, clients_id_list, seed=1234):
    # torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    clientsDict = {i: [] for i in clients_id_list}
    class_data = {cls: [] for cls in classes}

    # Organize dataset by classes
    for cls, data in dataset:
        class_data[cls].append(data)

    N = batchSize // len(classes)

    for i in clients_id_list:
        clientsDict[i] = []
        for cls in classes:
            if len(class_data[cls]) >= N:
                selected_data = class_data[cls][:N]
                class_data[cls] = class_data[cls][N:]
                clientsDict[i].extend(zip([cls] * N, selected_data))
            else:
                # if the data is left is less than 'N' -> use it all
                selected_data = class_data[cls]
                class_data[cls] = []
                clientsDict[i].extend(zip([cls] * len(selected_data), selected_data))

    return clientsDict
