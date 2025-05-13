import torch
import torch.nn as nn


class testNN_wo_Softmax_5_layer(nn.Module):
    def __init__(self, outputClasses):
        super(testNN_wo_Softmax_5_layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # (B, 32, 32, 32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)   # (B, 32, 32, 32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (B, 64, 16, 16)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # (B, 128, 8, 8)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # (B, 256, 4, 4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # After 4 poolings: 32 -> 16 -> 8 -> 4 -> 4 (last pooling not applied after conv5)
        self.fc = nn.Linear(256 * 8 * 8, outputClasses)
        self._initialize_weights()

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
        
        # print(f"input init {x.size()}")  # (32, 3, 32, 32)
        x = self.conv1(x)
        # x = self.relu(x)        
        # print(f"output pool1: {x.size()}")  # (32, 32, 16, 16)
        
        x = self.conv2(x)
        # x = self.relu(x)

        x = self.conv3(x)
        # print(f"output conv3: {x.size()}")
        if torch.isnan(x).any():
            print("Conv3 출력에 NaN이 있습니다.")
        x = self.relu(x)
        x = self.pool(x)
        # print(f"output pool3: {x.size()}")

        x = self.conv4(x)
        # print(f"output conv4: {x.size()}")
        if torch.isnan(x).any():
            print("Conv4 출력에 NaN이 있습니다.")
        x = self.relu(x)
        x = self.pool(x)
        # print(f"output pool4: {x.size()}")

        x = self.conv5(x)
        # print(f"output conv5: {x.size()}")
        if torch.isnan(x).any():
            print("Conv5 출력에 NaN이 있습니다.")
        x = self.relu(x)
        # No pooling after conv5

        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x