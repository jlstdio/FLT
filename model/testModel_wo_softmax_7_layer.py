import torch
import torch.nn as nn


class testNN_wo_Softmax_7_layer(nn.Module):
    def __init__(self, outputClasses):
        super(testNN_wo_Softmax_7_layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 출력: 32 x 32 x 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 출력: 64 x 32 x 32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 출력: 128 x 32 x 32
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 출력: 128 x 32 x 32
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 출력: 256 x 32 x 32
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 출력: 128 x 32 x 32
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # 출력: 256 x 32 x 32
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 출력 크기 절반으로 줄임
        self.relu = nn.ReLU()

        # 최종 출력 크기: (256 x 4 x 4) = 4096 (32x32 입력이 두 번 풀링되면 8x8으로 축소됨)
        self.fc = nn.Linear(1024 * 8 * 8, outputClasses)

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
        x = self.relu(self.conv1(x))
        if torch.isnan(x).any():
            print("Conv1 출력에 NaN이 있습니다.")
        x = self.pool(x)  # 출력 크기 절반 감소 (16x16)

        x = self.relu(self.conv2(x))
        if torch.isnan(x).any():
            print("Conv2 출력에 NaN이 있습니다.")
        x = self.pool(x)  # 출력 크기 절반 감소 (8x8)

        x = self.relu(self.conv3(x))
        if torch.isnan(x).any():
            print("Conv3 출력에 NaN이 있습니다.")

        x = self.relu(self.conv4(x))
        if torch.isnan(x).any():
            print("Conv4 출력에 NaN이 있습니다.")

        x = self.relu(self.conv5(x))
        if torch.isnan(x).any():
            print("Conv5 출력에 NaN이 있습니다.")

        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
