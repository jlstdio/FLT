import numpy as np


def partial_dirichlet_split(dataset, num_classes, alpha, num_clients):
    """
    CIFAR-10 데이터를 Dirichlet 분포를 사용하여 10개의 클라이언트로 분배합니다.
    첫 번째 클라이언트는 alpha=0.05로 강하게 non-IID하게 분배되고,
    나머지 9개의 클라이언트는 alpha=1.0으로 거의 IID하게 분배됩니다.

    Parameters:
    - dataset: 리스트 또는 배열 형태의 데이터셋, 각 요소는 (레이블, 데이터) 형태
    - num_classes: 클래스의 수 (CIFAR-10의 경우 10)
    - alpha: 나머지 9개 클라이언트에 적용할 Dirichlet alpha 값 (예: 1.0)
    - num_clients: 클라이언트의 수 (CIFAR-10의 경우 10)

    Returns:
    - dict_users: 각 클라이언트에 할당된 데이터의 딕셔너리
    """
    # 데이터셋 언패킹
    ys, xs = zip(*dataset)
    labels = np.array(ys)

    dict_users = {}
    multinomial_vals = []
    examples_per_label = []

    # 각 클래스별 예제 수 계산
    for i in range(num_classes):
        examples_per_label.append(np.sum(labels == i))

    # 각 클라이언트마다 Dirichlet 분포에서 비율 추출
    for i in range(num_clients):
        if i == 0:
            current_alpha = 0.05  # 첫 번째 클라이언트는 non-IID
        else:
            current_alpha = alpha    # 나머지 클라이언트는 거의 IID
        proportion = np.random.dirichlet(current_alpha * np.ones(len(num_classes)))
        multinomial_vals.append(proportion)

    multinomial_vals = np.array(multinomial_vals)
    example_indices = []

    # 각 클래스별로 예제 인덱스 섞기
    for k in range(num_classes):
        label_k_indices = np.where(labels == k)[0]
        np.random.shuffle(label_k_indices)
        example_indices.append(label_k_indices)

    example_indices = np.array(example_indices, dtype=object)

    client_samples = [[] for _ in range(num_clients)]
    count = np.zeros(len(num_classes)).astype(int)
    class_labels_for_clients = [[] for _ in range(num_clients)]

    examples_per_client = int(len(labels) / num_clients)

    # 클라이언트별로 예제를 분배
    for client in range(num_clients):
        for _ in range(examples_per_client):
            if multinomial_vals[client].sum() > 0:
                sampled_label = np.argmax(
                    np.random.multinomial(1, multinomial_vals[client] / multinomial_vals[client].sum())
                )
                label_indices = example_indices[sampled_label]
                if count[sampled_label] < examples_per_label[sampled_label]:
                    client_samples[client].append(xs[label_indices[count[sampled_label]]])  # 데이터 추가
                    class_labels_for_clients[client].append(sampled_label)
                    count[sampled_label] += 1

                    # 모든 클래스의 예제가 분배되면 해당 클래스의 비율을 0으로 설정
                    if count[sampled_label] == examples_per_label[sampled_label]:
                        multinomial_vals[:, sampled_label] = 0

    # 각 클라이언트의 샘플을 섞기
    for client in range(num_clients):
        paired_samples = list(zip(class_labels_for_clients[client], client_samples[client]))
        np.random.shuffle(paired_samples)
        dict_users[client] = paired_samples

    return dict_users
