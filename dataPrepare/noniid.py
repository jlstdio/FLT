import json
from collections import defaultdict
import numpy as np
import random


def dirichletSplit(dataset, classes, numClients, configPath, seed=1234):
    np.random.seed(seed)
    random.seed(seed)

    with open(configPath, 'r') as file:
        config = json.load(file)

    config = config['clientsType'][0]
    alpha = config['alpha']

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


def dirichlet_equal_split(dataset, classes, alpha, num_clients, seed):
    np.random.seed(seed)
    random.seed(seed)

    # Unzipping the dataset
    ys, xs = zip(*dataset)
    labels = np.array(ys)

    dict_users = {}
    multinomial_vals = []
    examples_per_label = []

    # Counting examples per class
    for i in classes:
        examples_per_label.append(np.sum(labels == i))

    # Each client has a multinomial distribution over classes drawn from a Dirichlet distribution
    for i in range(num_clients):
        proportion = np.random.dirichlet(alpha * np.ones(len(classes)))
        multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)
    example_indices = []

    # Shuffling examples for each class
    for k in classes:
        label_k_indices = np.where(labels == k)[0]
        np.random.shuffle(label_k_indices)
        example_indices.append(label_k_indices)

    example_indices = np.array(example_indices, dtype=object)

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(len(classes)).astype(int)
    class_labels_for_clients = [[] for _ in range(num_clients)]

    examples_per_client = int(len(labels) / num_clients)

    # Distributing examples to clients based on multinomial distribution
    for client in range(num_clients):
        for _ in range(examples_per_client):
            if multinomial_vals[client].sum() > 0:
                sampled_label = np.argmax(np.random.multinomial(1, multinomial_vals[client] / multinomial_vals[client].sum()))
                label_indices = example_indices[sampled_label]
                if count[sampled_label] < examples_per_label[sampled_label]:
                    client_samples[client].append(xs[label_indices[count[sampled_label]]]) # Append data not just index
                    class_labels_for_clients[client].append(sampled_label)
                    count[sampled_label] += 1

                    # Resetting probabilities when all examples of a class have been distributed
                    if count[sampled_label] == examples_per_label[sampled_label]:
                        multinomial_vals[:, sampled_label] = 0

    # Shuffling samples for each client
    for client in range(num_clients):
        paired_samples = list(zip(class_labels_for_clients[client], client_samples[client]))
        np.random.shuffle(paired_samples)
        dict_users[client] = paired_samples

    return dict_users


def pathologicalSplit(dataset, classes, numClients, configPath='', seed=1234):
    np.random.seed(seed)
    random.seed(seed)

    with open(configPath, 'r') as file:
        config = json.load(file)

    config = config['clientsType'][0]
    classesPerClient = int(config['classesPerClient'])

    # Initialize dictionary for clients
    clientsDict = {i: [] for i in range(numClients)}

    # Organize data by class
    class_data = {cls: [] for cls in classes}
    for cls, data in dataset:
        class_data[cls].append(data)

    # Shuffle classes to ensure random assignment
    shuffled_classes = np.random.permutation(classes)
    num_classes = len(shuffled_classes)

    # Calculate classes per client
    # Ensure that all classes are assigned
    if classesPerClient * numClients < num_classes:
        raise ValueError("classesPerClient * numClients must be >= number of classes")

    # Assign classes to clients
    client_classes = defaultdict(list)
    for idx, cls in enumerate(shuffled_classes):
        client_id = idx % numClients
        client_classes[client_id].append(cls)

    # Optionally, assign additional classes if classesPerClient > classes assigned
    for client_id in range(numClients):
        while len(client_classes[client_id]) < classesPerClient:
            additional_class = np.random.choice(shuffled_classes)
            if additional_class not in client_classes[client_id]:
                client_classes[client_id].append(additional_class)

    # Assign data to clients based on their assigned classes
    for client_id, assigned_classes in client_classes.items():
        for cls in assigned_classes:
            client_data = class_data[cls]
            clientsDict[client_id].extend([(cls, data) for data in client_data])

    return clientsDict