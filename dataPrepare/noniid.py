import numpy as np

def dirichletSplit(dataset, classes, alpha, numClients):
    clientsDict = {i: [] for i in range(numClients)}
    class_data = {cls: [] for cls in classes}

    for cls, data in dataset:
        class_data[cls].append(data)

    for cls in classes:
        # dirichlet distribution
        class_distribution = np.random.dirichlet([alpha] * numClients)

        # Calculate the number of data points for each client
        num_class_data = len(class_data[cls])
        class_data_idxs = np.arange(num_class_data)
        np.random.shuffle(class_data_idxs)

        class_data_per_client = (class_distribution * num_class_data).astype(int)

        start_idx = 0
        for client_id in range(numClients):
            num_data = class_data_per_client[client_id]
            end_idx = start_idx + num_data
            selected_data_idxs = class_data_idxs[start_idx:end_idx]
            selected_data = [class_data[cls][i] for i in selected_data_idxs]
            clientsDict[client_id].extend(zip([cls] * num_data, selected_data))
            start_idx = end_idx

        # If there are leftover data points, distribute them to clients
        leftover_data_idxs = class_data_idxs[start_idx:]
        if leftover_data_idxs.size > 0:
            leftover_data = [class_data[cls][i] for i in leftover_data_idxs]
            for idx, data in enumerate(leftover_data):
                clientsDict[idx % numClients].append((cls, data))

    return clientsDict
