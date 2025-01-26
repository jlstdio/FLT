import numpy as np
import torch
import torch.nn as nn


class testNN_wo_Softmax_more_filters(nn.Module):
    def __init__(self, outputClasses):
        super(testNN_wo_Softmax_more_filters, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # Batch Normalization after Conv1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)  # Batch Normalization after Conv2
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)  # Batch Normalization after Conv3
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256 * 32 * 32, outputClasses)  # Fully connected layer

        self._initialize_weights()  # He initialization

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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # BatchNorm after Conv1
        if torch.isnan(x).any():
            print("Conv1 + BatchNorm1 출력에 NaN이 있습니다.")
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)  # BatchNorm after Conv2
        if torch.isnan(x).any():
            print("Conv2 + BatchNorm2 출력에 NaN이 있습니다.")
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)  # BatchNorm after Conv3
        if torch.isnan(x).any():
            print("Conv3 + BatchNorm3 출력에 NaN이 있습니다.")
        x = self.relu(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
