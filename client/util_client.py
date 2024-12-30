from collections import defaultdict

import torch
from torch import nn


def target_type_convert(costFunc, targets):
    targets_converted = None

    if costFunc == 'CEloss':
        targets_converted = targets.long()
    elif costFunc == 'BCEloss':
        targets_converted = targets
    elif costFunc == 'BCEWithLogitsLoss':
        targets_converted = targets.long()

    return targets_converted


def criterion_select(costFunc):
    criterion = None

    if costFunc == 'CEloss':
        criterion = nn.CrossEntropyLoss()
    elif costFunc == 'BCEloss':
        criterion = nn.BCELoss()
    elif costFunc == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()

    return criterion


def clip_implement(costFunc, model, normClip):
    if costFunc == 'CEloss':
        pass
    elif costFunc == 'BCEloss':
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=normClip)
    elif costFunc == 'BCEWithLogitsLoss':
        pass


def split_data_by_class_ratio(dataset, train_ratio):
    data_by_class = defaultdict(list)
    train_classes = []
    train_data = []
    test_classes = []
    test_data = []

    # 입력된 데이터셋을 클래스별로 분류
    for cls, data in dataset:
        data_by_class[cls].append(data)

    # 각 클래스별로 train과 test로 분할
    for cls, data in data_by_class.items():
        n_total = len(data)
        n_test = int(n_total * (1.0 - train_ratio))

        # 테스트 데이터의 수를 최소 0으로 설정
        n_test = max(n_test, 0)

        if n_test > 0:
            test_data_cls = data[:n_test]
            train_data_cls = data[n_test:]

            # 테스트 데이터 추가
            test_classes.extend([cls] * len(test_data_cls))
            test_data.extend(test_data_cls)

            # 트레인 데이터 추가
            train_classes.extend([cls] * len(train_data_cls))
            train_data.extend(train_data_cls)
        else:
            # 테스트 데이터가 없는 경우 트레인 데이터에만 추가
            train_classes.extend([cls] * len(data))
            train_data.extend(data)

    # 트레인과 테스트 데이터를 zip 객체로 변환
    train_zip = zip(train_classes, train_data)
    test_zip = zip(test_classes, test_data)

    return train_zip, test_zip


