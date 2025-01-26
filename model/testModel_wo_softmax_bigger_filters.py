import numpy as np
import torch
import torch.nn as nn


class testNN_wo_Softmax_bigger_filters(nn.Module):
    def __init__(self, outputClasses):
        super(testNN_wo_Softmax_bigger_filters, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)  # Conv1 BatchNorm
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)  # Conv2 BatchNorm

        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)  # Conv3 BatchNorm

        self.relu = nn.ReLU()
        self.fc = nn.Linear(64 * 32 * 32, outputClasses)  # 64 채널, 마지막 feature map의 크기 8*8

        self._initialize_weights()  # He 초기화 함수 호출

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # BatchNorm 적용
        if torch.isnan(x).any():
            print("Conv1 출력에 NaN이 있습니다.")
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)  # BatchNorm 적용
        if torch.isnan(x).any():
            print("Conv2 출력에 NaN이 있습니다.")
        # x = self.pool(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)  # BatchNorm 적용
        if torch.isnan(x).any():
            print("Conv3 출력에 NaN이 있습니다.")
        x = self.relu(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
