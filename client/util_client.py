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