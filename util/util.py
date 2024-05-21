import matplotlib.pyplot as plt


def showDistribution(clientsDict, classes):
    num_clients = len(clientsDict)
    class_counts = {i: {cls: 0 for cls in classes} for i in range(num_clients)}

    # 각 클라이언트의 클래스별 데이터 개수 계산
    for client, data in clientsDict.items():
        for cls, _ in data:
            class_counts[client][cls] += 1

    # 플롯 그리기
    fig, axes = plt.subplots(1, num_clients, figsize=(15, 5), sharey=True)
    if num_clients == 1:
        axes = [axes]

    for client in range(num_clients):
        counts = [class_counts[client][cls] for cls in classes]
        axes[client].bar(classes, counts)
        axes[client].set_title(f'Client {client}')
        axes[client].set_xlabel('Class')
        axes[client].set_ylabel('Count')

    plt.tight_layout()
    plt.show()